from os.path import join, isdir, exists
from os import listdir, mkdir, makedirs
from tqdm import tqdm
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from concurrent import futures
import sys
import time
from glob import glob

# Print iterations progress (thanks StackOverflow)
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0), nearest=False):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float32)
    # Masks are labels rather than intensities, so they are resampled without
    # interpolation to stay binary
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), flags=flags,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instance_size=255, padding=(0, 0, 0), mask=None):
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    target_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instance_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instance_size, padding)

    if mask is None:
        return z, x, None

    # The same geometry as x, so the mask stays registered with the search
    # crop the loss compares it against
    x_mask = crop_hwc(mask, pos_s_2_bbox(target_pos, s_x), instance_size,
                      padding=0, nearest=True)
    return z, x, x_mask


def crop_video(video, image_folder, crop_path, instance_size):
    def get_im_num(line, adj=0):
        return int(line[2])

    def get_bbox(line):
        return [int(float(i)) for i in line[3:7]]

    def get_mask(line, im_shape, bbox):
        """Full frame mask for this annotation, or None.

        The tenth column names a mask stored cropped to the bounding box,
        which is kwiver's convention. Placing it back into a frame sized
        canvas is what lets it go through the same crop as the image.
        """
        if len(line) < 10 or not line[9]:
            return None

        mask_file = join(image_folder, video, line[9])

        if not exists(mask_file):
            return None

        patch = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        if patch is None:
            return None

        canvas = np.zeros(im_shape[:2], dtype=np.uint8)

        x1 = max(bbox[0], 0)
        y1 = max(bbox[1], 0)
        x2 = min(bbox[0] + patch.shape[1], im_shape[1])
        y2 = min(bbox[1] + patch.shape[0], im_shape[0])

        if x2 <= x1 or y2 <= y1:
            return None

        canvas[y1:y2, x1:x2] = patch[:y2-y1, :x2-x1]
        return canvas

    def box_overlap(b1, b2):
        x_left = max( b1[0], b2[0] )
        y_top = max( b1[1], b2[1] )
        x_right = min( b1[2], b2[2] )
        y_bottom = min( b1[3], b2[3] )

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection_area = ( x_right - x_left ) * ( y_bottom - y_top )
        bb1_area = ( b1[2] - b1[0] ) * ( b1[3] - b1[1] )
        if bb1_area == 0:
            return 0.0

        return float( intersection_area ) / bb1_area

    video_crop_base_path = join(crop_path, video)
    if not isdir(video_crop_base_path): makedirs(video_crop_base_path)
    csv = glob(join(image_folder, video, '*.csv'))
    assert len(csv) == 1, "Only CSV in the folder must be the ground truth."
    csv = csv[0]
    with open(csv) as f:
        csv_lines = f.readlines()
    csv_lines = [line.strip('\n').split(',') for line in csv_lines if '#' not in line]

    track_nums = sorted(list(set([x[0] for x in csv_lines])))
    tracks = {}
    for track in track_nums: tracks[track] = []
    for line in csv_lines:
        tracks[line[0]].append(line)
    for track in track_nums: tracks[track] = sorted(tracks[track], key=lambda x: x[1])
    min_im = min([get_im_num(tracks[track][0]) for track in track_nums])
    max_im = max([get_im_num(tracks[track][-1]) for track in track_nums])
    image_files = []
    common_ext = [ "*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG" ]
    for ext in common_ext:
        image_files.extend(glob(join(image_folder, video, ext)))
    image_files = sorted(image_files)
    for idx, track in enumerate(track_nums):
        for line in tqdm(tracks[track]):
            im_num = get_im_num(line, min_im)
            bbox = get_bbox(line)
            image_file = join(image_folder, video, line[1])
            if not exists(image_file) and len(image_files) > int(line[2]):
                image_file = image_files[int(line[2])]
            im = cv2.imread(image_file)
            assert not im is None, "Missing image."
            if box_overlap(bbox, [ 0, 0, im.shape[0], im.shape[1] ]) < 0.50:
                continue
            avg_chans = np.mean(im, axis=(0, 1))
            mask = get_mask(line, im.shape, bbox)
            z, x, x_mask = crop_like_SiamFC(im, bbox, instance_size=instance_size, padding=avg_chans, mask=mask)
            z_path = join(video_crop_base_path, f'{im_num:08}.{idx:08}.z.jpg')
            x_path = join(video_crop_base_path, f'{im_num:08}.{idx:08}.x.jpg')
            cv2.imwrite(x_path, x)
            cv2.imwrite(z_path, z)
            if x_mask is not None:
                # Named off the search crop so the dataset can find it without
                # dataset.json having to carry it
                cv2.imwrite(join(video_crop_base_path,
                                 f'{im_num:08}.{idx:08}.x.mask.png'), x_mask)


def par_crop(instance_size=511, num_threads=24, image_folder='data_folder', save_folder='siamrpn++_model'):
    dataDir = '.'
    crop_path = join(save_folder, 'crop{:d}'.format(instance_size))
    if not isdir(crop_path): makedirs(crop_path)
    videos = [x.split('/')[-2] for x in sorted(glob(join(image_folder, '*', '*.csv')))]
    n_videos = len(videos)
    with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        fs = [executor.submit(crop_video, video, image_folder, crop_path, instance_size) for video in videos]
        for i, f in enumerate(futures.as_completed(fs)):
           printProgress(i, n_videos, suffix='Done ', barLength=40)
