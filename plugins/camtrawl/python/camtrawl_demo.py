# -*- coding: utf-8 -*-
"""
Runs camtrawl algos with no dependency on kwiver
"""
from __future__ import division, print_function, unicode_literals
from imutils import (imscale, overlay_heatmask, putMultiLineText)
from os.path import expanduser, basename, join
from functools import partial
import glob
import cv2
import numpy as np
import ubelt as ub
import threading
import camtrawl_algos as ctalgo

try:
    import queue
except ImportError:
    # python 2 support
    import Queue as queue


class DrawHelper(object):
    """
    Visualization of the algorithm stages
    """

    @staticmethod
    def draw_detections(img, detections, masks, assigned={}):
        """
        Draws heatmasks showing where detector was triggered.
        Bounding boxes and contours are drawn over accepted detections.
        """
        # Upscale masks to original image size
        dsize = tuple(img.shape[0:2][::-1])
        shape = img.shape[0:2]

        # Align all masks with the image size
        masks2 = {
            k: cv2.resize(v, dsize, interpolation=cv2.INTER_NEAREST)
            for k, v in masks.items()
        }

        # Create detection heatmap
        draw_mask = np.zeros(shape, dtype=np.float)

        if 'orig' in masks2:
            draw_mask[masks2['orig'] > 0] = .4
        if 'local' in masks2:
            draw_mask[masks2['local'] > 0] = .65
        if 'post' in masks2:
            draw_mask[masks2['post'] > 0] = .85

        for n, detection in enumerate(detections, start=1):
            bbox = detection.bbox
            dmask, factor = imscale(detection.mask.astype(np.uint8),
                                    detection.bbox_factor)
            yslice = slice(bbox.ymin, bbox.ymax + int(detection.bbox_factor))
            xslice = slice(bbox.xmin, bbox.xmax + int(detection.bbox_factor))
            draw_mask[yslice, xslice][dmask > 0] = 1.0

        draw_img = overlay_heatmask(img, draw_mask, alpha=.7)
        BGR_GREEN = (0, 255, 0)
        BGR_BLUE = (255, 0, 0)
        # BGR_RED = (0, 0, 255)
        BGR_PURPLE = (255, 0, 255)

        # Draw bounding boxes and contours
        for i, detection in enumerate(detections):
            # Points come back in (x, y), but we want to draw in (r, c)
            box_points = np.round(detection.box_points()).astype(np.int)
            hull_points = detection.hull()
            hull_color = BGR_BLUE
            bbox_color = BGR_GREEN if i in assigned else BGR_PURPLE
            draw_img = cv2.drawContours(
                image=draw_img, contours=[hull_points], contourIdx=-1,
                color=hull_color, thickness=2)
            draw_img = cv2.drawContours(
                image=draw_img, contours=[box_points], contourIdx=-1,
                color=bbox_color, thickness=2)
        return draw_img

    @staticmethod
    def draw_stereo_detections(img1, detections1, masks1,
                               img2, detections2, masks2,
                               assignment=None, assign_data=None,
                               cand_errors=None):
        """
        Draws stereo detections side-by-side and draws lines indicating
        assignments (and near-assignments for debugging)
        """
        import textwrap
        BGR_RED = (0, 0, 255)
        BGR_PURPLE = (255, 0, 255)

        accepted_color = BGR_RED
        rejected_color = BGR_PURPLE

        assigned1 = {} if assignment is None else [t[0] for t in assignment]
        assigned2 = {} if assignment is None else [t[1] for t in assignment]
        draw1 = DrawHelper.draw_detections(img1, detections1, masks1,
                                           assigned=assigned1)
        draw2 = DrawHelper.draw_detections(img2, detections2, masks2,
                                           assigned=assigned2)
        stacked = np.hstack([draw1, draw2])

        if assignment is not None and len(cand_errors) > 0:

            # Draw candidate assignments that did not pass the thresholds
            for j in range(cand_errors.shape[1]):
                i = np.argmin(cand_errors[:, j])
                if (i, j) in assignment or (j, i) in assignment:
                    continue
                error = cand_errors[i, j]

                detection1 = detections1[i]
                detection2 = detections2[j]
                center1 = np.array(detection1.oriented_bbox().center)
                center2 = np.array(detection2.oriented_bbox().center)

                # Offset center2 to the right image
                center2_ = center2 + [draw1.shape[1], 0]

                center1 = tuple(center1.astype(np.int))
                center2_ = tuple(center2_.astype(np.int))

                stacked = cv2.line(stacked, center1, center2_,
                                   color=rejected_color, lineType=cv2.LINE_AA,
                                   thickness=1)
                text = textwrap.dedent(
                    '''
                    error = {error:.2f}
                    '''
                ).strip().format(error=error)

                stacked = putMultiLineText(stacked, text, org=center2_,
                                           fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=1.5, color=rejected_color,
                                           thickness=2, lineType=cv2.LINE_AA)

            # Draw accepted assignments
            for (i, j), info in zip(assignment, assign_data):
                detection1 = detections1[i]
                detection2 = detections2[j]
                center1 = np.array(detection1.oriented_bbox().center)
                center2 = np.array(detection2.oriented_bbox().center)

                # Offset center2 to the right image
                center2_ = center2 + [draw1.shape[1], 0]

                center1 = tuple(center1.astype(np.int))
                center2_ = tuple(center2_.astype(np.int))

                stacked = cv2.line(stacked, center1, center2_,
                                   color=accepted_color, lineType=cv2.LINE_AA,
                                   thickness=2)

                text = textwrap.dedent(
                    '''
                    len = {fishlen:.2f}mm
                    error = {error:.2f}
                    range = {range:.2f}mm
                    '''
                ).strip().format(**info)

                stacked = putMultiLineText(stacked, text, org=center1,
                                           fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=1.5, color=accepted_color,
                                           thickness=2, lineType=cv2.LINE_AA)

        with_orig = False
        if with_orig:
            # Put in the original images
            bottom = np.hstack([img1, img2])
            stacked = np.vstack([stacked, bottom])

        return stacked


class FrameStream(ub.NiceRepr):
    """
    Helper for iterating through a sequence of image frames.
    This will be replaced by KWIVER.
    """
    def __init__(self, image_path_list, stride=1):
        self.image_path_list = image_path_list
        self.stride = stride
        self.length = len(image_path_list)

    def __nice__(self):
        return 'nframes={}'.format(len(self))

    def __len__(self):
        return self.length

    def _parse_frame_id(self, img_fpath):
        frame_id = basename(img_fpath).split('_')[0]
        return frame_id

    def __getitem__(self, index):
        img_fpath = self.image_path_list[index]
        frame_id = self._frame_id(img_fpath)
        img = cv2.imread(img_fpath)
        return frame_id, img

    def __iter__(self):
        for i in range(0, len(self), self.stride):
            yield self[i]


def buffered_imread(image_path_list, buffer_size=2):
    """
    Generator that runs imread in a separate thread.
    The next image is read while you are processing the current one.

    Args:
        buffer_size: the maximal number of items to pre-generate (length of the
            buffer)

    Note:
        Note: use threads instead of processes because the GIL should not
        prevent IO-bound functions like cv2.imread from running in parallel, so
        we can be reading an image while we are processing the previous one.
        (also processes have to serialize objects, which defeats the purpose of
        reading them in parallel)

    References:
        # Adapted from
        https://github.com/benanne/kaggle-ndsb/blob/11a66cdbddee16c69514b9530a727df0ac6e136f/buffering.py
    """
    if buffer_size < 2:
        raise ValueError('Minimal buffer size is 2')

    # the effective buffer size is one less, because the generation process
    # will generate one extra element and block until there is room in the buffer.
    buffer = queue.Queue(maxsize=buffer_size - 1)

    thread = threading.Thread(target=_imreader_thread,
                              args=(image_path_list, buffer))
    thread.daemon = True
    thread.start()

    # Generate the images
    data = buffer.get()
    if data is not None:
        yield data
    while data is not None:
        data = buffer.get()
        yield data


def serial_imread(image_path_list):
    for fpath in image_path_list:
        yield cv2.imread(fpath)


def _imreader_thread(image_path_list, buffer):
    for fpath in image_path_list:
        data = cv2.imread(fpath)
        # block=True means wait until there is room on the queue
        buffer.put(data, block=True)
    buffer.put(None)  # sentinel: signal the end of the iterator


class StereoFrameStream(object):
    """
    Exmaple:
        >>> import sys
        >>> sys.path.append('/home/joncrall/code/VIAME/plugins/camtrawl/python')
        >>> from camtrawl_demo import *
        >>> img_path1, img_path2, cal = demodata_input(dataset='haul83')
        >>> self = StereoFrameStream(img_path1, img_path2)
        >>> self.preload()
        >>> # -------
        >>> prog_thread = ub.ProgIter(self._stream(buffer_size=2), label='thread', length=len(self), adjust=False, freq=1)
        >>> for _ in prog_thread:
        >>>     pass
        >>> print('prog_thread._total_seconds = {!r}'.format(prog_thread._total_seconds))
        >>> # -------
        >>> prog_serial = ub.ProgIter(self._stream(buffer_size=None), label='serial', length=len(self), adjust=False, freq=1)
        >>> for _ in prog_serial:
        >>>     pass
        >>> print('prog_serial._total_seconds = {!r}'.format(prog_serial._total_seconds))
    """
    def __init__(self, img_path1, img_path2):
        self.img_path1 = img_path1
        self.img_path2 = img_path2
        self.image_path_list1 = None
        self.image_path_list2 = None
        self.aligned_frameids = []
        self.aligned_idx1 = None
        self.aligned_idx2 = None
        self.index = 0
        self.buffer_size = 2

    def seek(self, index):
        self.index = index

    def __len__(self):
        return len(self.aligned_frameids) - self.index

    def _stream(self, buffer_size=None):
        if buffer_size is None:
            buffer_size = self.buffer_size
        if buffer_size and buffer_size > 1:
            buffer_func = partial(buffered_imread, buffer_size=buffer_size)
        else:
            buffer_func = serial_imread

        idx1_slice = self.aligned_idx1[self.index:]
        idx2_slice = self.aligned_idx2[self.index:]
        frame_ids = self.aligned_frameids[self.index:]

        fpath_gen1 = ub.take(self.image_path_list1, idx1_slice)
        fpath_gen2 = ub.take(self.image_path_list2, idx2_slice)
        img_gen1 = buffer_func(fpath_gen1)
        img_gen2 = buffer_func(fpath_gen2)

        for frame_id, img1, img2 in zip(frame_ids, img_gen1, img_gen2):
            yield frame_id, img1, img2
            self.index += 1

    def __iter__(self):
        for data in self._stream():
            yield data

    def _parse_frame_id(self, img_fpath):
        frame_id = basename(img_fpath).split('_')[0]
        return frame_id

    def preload(self):
        self.image_path_list1 = sorted(glob.glob(join(self.img_path1, '*.jpg')))
        self.image_path_list2 = sorted(glob.glob(join(self.img_path2, '*.jpg')))
        self.align_frames()

    def align_frames(self, verbose=1):
        frame_ids1 = list(map(self._parse_frame_id, self.image_path_list1))
        frame_ids2 = list(map(self._parse_frame_id, self.image_path_list2))

        n_imgs1 = len(self.image_path_list1)
        n_imgs2 = len(self.image_path_list2)

        aligned_idx1 = []
        aligned_idx2 = []
        aligned_frameids = []

        idx1, idx2 = 0, 0
        while idx1 < n_imgs1 and idx2 < n_imgs2:
            frame_id1 = frame_ids1[idx1]
            frame_id2 = frame_ids2[idx2]
            if frame_id1 == frame_id2:
                aligned_idx1.append(idx1)
                aligned_idx2.append(idx2)
                aligned_frameids.append(frame_id1)
                idx1 += 1
                idx2 += 1
            elif frame_id1 < frame_id2:
                idx1 += 1
            elif frame_id1 > frame_id2:
                idx2 += 1

        if verbose:
            # missing_from1 = set(frame_ids2) - set(frame_ids1)
            # missing_from2 = set(frame_ids1) - set(frame_ids2)
            # print('missing_from1 = {!r}'.format(sorted(missing_from1)))
            # print('missing_from2 = {!r}'.format(sorted(missing_from2)))
            n_aligned = len(aligned_idx1)
            print('Camera1 aligned {}/{} frames'.format(n_aligned, n_imgs1))
            print('Camera2 aligned {}/{} frames'.format(n_aligned, n_imgs2))

        self.aligned_frameids = aligned_frameids
        self.aligned_idx1 = aligned_idx1
        self.aligned_idx2 = aligned_idx2


def demodata_input(dataset='test'):
    """
    Specifies the input files for testing and demos
    """
    if dataset == 'test':
        data_fpath = expanduser('~/data/autoprocess_test_set')
        cal_fpath = join(data_fpath, 'cal_201608.mat')
        img_path1 = join(data_fpath, 'image_data/left')
        img_path2 = join(data_fpath, 'image_data/right')
    elif dataset == 'haul83-small':
        data_fpath = expanduser('~/data/camtrawl_stereo_sample_data/small')
        cal_fpath = join(data_fpath, '201608_calibration_data/selected/Camtrawl_2016.npz')
        img_path1 = join(data_fpath, 'Haul_83/left')
        img_path2 = join(data_fpath, 'Haul_83/right')
    elif dataset == 'haul83':
        data_fpath = expanduser('~/data/camtrawl_stereo_sample_data/')
        cal_fpath = join(data_fpath, '201608_calibration_data/selected/Camtrawl_2016.npz')
        img_path1 = join(data_fpath, 'Haul_83/D20160709-T021759/images/AB-800GE_00-0C-DF-06-40-BF')  # left
        img_path2 = join(data_fpath, 'Haul_83/D20160709-T021759/images/AM-800GE_00-0C-DF-06-20-47')  # right
    else:
        raise ValueError('unknown dataset={!r}'.format(dataset))
    return img_path1, img_path2, cal_fpath


def demodata_detections(dataset='haul83', target_step='detect', target_frame_num=7):
    """
    Helper for doctests. Gets test data at different points in the pipeline.
    """
    # <ipython hacks>
    if 'target_step' not in vars():
        target_step = 'detect'
    if 'target_frame_num' not in vars():
        target_frame_num = 7
    # </ipython hacks>
    img_path1, img_path2, cal_fpath = demodata_input(dataset=dataset)

    stream = StereoFrameStream(img_path1, img_path2)
    stream.preload()

    cal = ctalgo.StereoCalibration.from_file(cal_fpath)

    detector1 = ctalgo.GMMForegroundObjectDetector()
    detector2 = ctalgo.GMMForegroundObjectDetector()

    for frame_num, (frame_id, img1, img2) in enumerate(stream):
        if frame_num == target_frame_num:
            if target_step == 'detect':
                return detector1, img1

        detections1 = detector1.detect(img1)
        detections2 = detector2.detect(img2)
        masks1 = detector1._masks
        masks2 = detector2._masks

        n_detect1, n_detect2 = len(detections1), len(detections2)
        print('frame_num, (n_detect1, n_detect2) = {} ({}, {})'.format(
            frame_num, n_detect1, n_detect2))

        if frame_num == target_frame_num:
            stacked = DrawHelper.draw_stereo_detections(img1, detections1, masks1,
                                                        img2, detections2, masks2)
            drawing_dpath = ub.ensuredir('out')
            cv2.imwrite(drawing_dpath + '/mask{}_draw.png'.format(frame_id), stacked)
            break

    return detections1, detections2, cal


def demo():
    """
    Runs the algorithm end-to-end.
    """
    # dataset = 'test'
    dataset = 'haul83'

    img_path1, img_path2, cal_fpath = demodata_input(dataset=dataset)

    # ----
    # Choose parameter configurations
    # ----

    # Use GMM based model
    gmm_params = {
    }
    triangulate_params = {
    }

    DRAWING = 1

    # ----
    # Initialize algorithms
    # ----

    detector1 = ctalgo.GMMForegroundObjectDetector(**gmm_params)
    detector2 = ctalgo.GMMForegroundObjectDetector(**gmm_params)
    triangulator = ctalgo.FishStereoMeasurments(**triangulate_params)

    import pyfiglet
    print(pyfiglet.figlet_format('CAMTRAWL', font='cybermedium'))
    print('dataset = {!r}'.format(dataset))
    print('Detector1 Config: ' + ub.repr2(detector1.config, nl=1))
    print('Detector2 Config: ' + ub.repr2(detector2.config, nl=1))
    print('Triangulate Config: ' + ub.repr2(triangulator.config, nl=1))
    print('DRAWING = {!r}'.format(DRAWING))

    cal = ctalgo.StereoCalibration.from_file(cal_fpath)

    stream = StereoFrameStream(img_path1, img_path2)
    stream.preload()
    stream.seek(2200)

    # ----
    # Run the algorithm
    # ----

    # n_frames = 2000
    # stream.aligned_frameids = stream.aligned_frameids[:stream.index]

    prog = ub.ProgIter(stream, clearline=True, length=len(stream),
                       adjust=False, label='camtrawl')
    _iter = prog

    if DRAWING:
        drawing_dpath = ub.ensuredir('out_{}'.format(dataset))
        ub.delete(drawing_dpath)
        ub.ensuredir(drawing_dpath)

    measure_fpath = ub.ensuredir('measurements_{}.csv'.format(dataset))
    ub.delete(measure_fpath)
    output_file = open(measure_fpath, 'a')

    headers = ['current_frame', 'fishlen', 'range', 'error', 'dz', 'box_pts1',
               'box_pts2']
    output_file.write(','.join(headers) + '\n')
    output_file.flush()

    measurements = []

    print('begin iteration')
    for frame_num, (frame_id, img1, img2) in enumerate(_iter):

        detections1 = list(detector1.detect(img1))
        detections2 = list(detector2.detect(img2))
        masks1 = detector1._masks
        masks2 = detector2._masks

        any_detected = len(detections1) > 0 or len(detections2) > 0

        if any_detected:
            assignment, assign_data, cand_errors = triangulator.find_matches(
                cal, detections1, detections2)
            # Append assignments to the measurements
            for data in assign_data:
                data['current_frame'] = int(frame_id)
                measurements.append(data)
                line = ','.join([repr(d).replace('\n', ' ').replace(',', ';')
                                 for d in ub.take(data, headers)])
                output_file.write(line + '\n')
                output_file.flush()
        else:
            cand_errors = None
            assignment, assign_data = None, None

        if DRAWING >= 2 or (DRAWING and any_detected):
            DRAWING = 3
            stacked = DrawHelper.draw_stereo_detections(img1, detections1, masks1,
                                                        img2, detections2, masks2,
                                                        assignment, assign_data,
                                                        cand_errors)
            stacked = cv2.putText(stacked,
                                  text='frame #{}, id={}'.format(frame_num,
                                                                 frame_id),
                                  org=(10, 50),
                                  fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=1, color=(255, 0, 0),
                                  thickness=2, lineType=cv2.LINE_AA)
            cv2.imwrite(drawing_dpath + '/mask{}_draw.png'.format(frame_id), stacked)
    output_file.close()

    n_total = len(measurements)
    print('n_total = {!r}'.format(n_total))
    if n_total:
        all_errors = np.array([d['error'] for d in measurements])
        all_lengths = np.array([d['fishlen'] for d in measurements])
        print('ave_error = {:.2f} +- {:.2f}'.format(all_errors.mean(), all_errors.std()))
        print('ave_lengths = {:.2f} +- {:.2f} '.format(all_lengths.mean(), all_lengths.std()))
    return measurements


def to_mat_format():
    import pandas as pd
    measure_fpath = ub.ensuredir('measurements_haul83.csv')
    py_df = pd.DataFrame.from_csv(measure_fpath, index_col=None)
    py_df['fishlen'] = py_df['fishlen'] / 10
    bbox_pts1 = py_df['box_pts1'].map(lambda p: eval(p.replace(';', ','), np.__dict__))
    bbox_pts2 = py_df['box_pts2'].map(lambda p: eval(p.replace(';', ','), np.__dict__))

    bbox_pts1 = np.array(bbox_pts1.values.tolist())
    bbox_pts2 = np.array(bbox_pts2.values.tolist())

    X = bbox_pts1.T[0].T
    Y = bbox_pts1.T[1].T
    X = pd.DataFrame(X, columns=['LX1', 'LX2', 'LX3', 'LX4'])
    Y = pd.DataFrame(Y, columns=['LY1', 'LY2', 'LY3', 'LY4'])
    py_df.join(X.join(Y))

    X = bbox_pts2.T[0].T
    Y = bbox_pts2.T[1].T
    X = pd.DataFrame(X, columns=['RX1', 'RX2', 'RX3', 'RX4'])
    Y = pd.DataFrame(Y, columns=['RY1', 'RY2', 'RY3', 'RY4'])
    py_df = py_df.join(X.join(Y))

    py_df = py_df.rename(columns={
        'error': 'Err',
        'fishlen': 'fishLength',
        'range': 'fishRange',
    })
    py_df.drop(['box_pts1', 'box_pts2'], axis=1, inplace=True)
    py_df.to_csv('haul83_py_results.csv')
    pass


def compare_results():
    print('Comparing results')
    import pandas as pd
    from tabulate import tabulate
    measure_fpath = ub.ensuredir('measurements_haul83.csv')
    py_df = pd.DataFrame.from_csv(measure_fpath, index_col=None)
    py_df['fishlen'] = py_df['fishlen'] / 10
    py_df['box_pts1'] = py_df['box_pts1'].map(lambda p: eval(p.replace(';', ','), np.__dict__))
    py_df['box_pts2'] = py_df['box_pts2'].map(lambda p: eval(p.replace(';', ','), np.__dict__))

    py_df['obox1'] = [ctalgo.OrientedBBox(*cv2.minAreaRect(pts[:, None, :].astype(np.int))) for pts in py_df['box_pts1']]
    py_df['obox2'] = [ctalgo.OrientedBBox(*cv2.minAreaRect(pts[:, None, :].astype(np.int))) for pts in py_df['box_pts2']]
    py_df.drop(['box_pts1', 'box_pts2'], axis=1, inplace=True)

    py_df['current_frame'] = py_df['current_frame'].astype(np.int)
    py_df = py_df.rename(columns={
        'error': 'Err',
        'fishlen': 'fishLength',
        'range': 'fishRange',
    })

    mat_df = _read_kresimir_results()

    intersect_frames = np.intersect1d(mat_df.current_frame, py_df.current_frame)
    print('intersecting frames = {} / {} (matlab)'.format(len(intersect_frames), len(set(mat_df.current_frame))))
    print('intersecting frames = {} / {} (python)'.format(len(intersect_frames), len(set(py_df.current_frame))))

    min_assign = ctalgo.FishStereoMeasurments.minimum_weight_assignment

    correspond = []
    for f in intersect_frames:
        pidxs = np.where(py_df.current_frame == f)[0]
        midxs = np.where(mat_df.current_frame == f)[0]

        pdf = py_df.iloc[pidxs]
        mdf = mat_df.iloc[midxs]

        ppts1 = np.array([o.center for o in pdf['obox1']])
        mpts1 = np.array([o.center for o in mdf['obox1']])

        ppts2 = np.array([o.center for o in pdf['obox2']])
        mpts2 = np.array([o.center for o in mdf['obox2']])

        import sklearn.metrics
        dists1 = sklearn.metrics.pairwise.pairwise_distances(ppts1, mpts1)
        dists2 = sklearn.metrics.pairwise.pairwise_distances(ppts2, mpts2)
        thresh = 100
        for i, j in min_assign(dists1):
            d1 = dists1[i, j]
            d2 = dists2[i, j]
            if d1 < thresh and d2 < thresh and abs(d1 - d2) < thresh / 4:
                correspond.append((pidxs[i], midxs[j]))
    correspond = np.array(correspond)

    # pflags = np.array(ub.boolmask(correspond.T[0], len(py_df)))
    mflags = np.array(ub.boolmask(correspond.T[1], len(mat_df)))
    print('there are {} detections that seem to be in common'.format(len(correspond)))
    print('The QC flags of the common detections are:       {}'.format(ub.dict_hist(mat_df[mflags]['QC'].values)))
    print('The QC flags of the other matlab detections are: {}'.format(ub.dict_hist(mat_df[~mflags]['QC'].values)))

    min_frame = py_df.current_frame.min()
    max_frame = py_df.current_frame.max()

    py_df = py_df[(py_df.current_frame >= min_frame) & (py_df.current_frame <= max_frame)]
    mat_df = mat_df[(mat_df.current_frame >= min_frame) & (mat_df.current_frame <= max_frame)]

    print('All stats')
    stats = pd.DataFrame(columns=['python', 'matlab'])
    for key in ['fishLength', 'fishRange', 'Err']:
        stats.loc[key, 'python'] = '{:6.2f} ± {:6.2f}'.format(py_df[key].mean(), py_df[key].std())
        stats.loc[key, 'matlab'] = '{:6.2f} ± {:6.2f}'.format(mat_df[key].mean(), mat_df[key].std())

    stats.loc['nTotal', 'python'] = '{}'.format(len(py_df))
    stats.loc['nTotal', 'matlab'] = '{}'.format(len(mat_df))
    print(tabulate(stats, headers='keys', tablefmt='psql', stralign='right'))

    print('Only COMMON detections')
    py_df_c = py_df.iloc[correspond.T[0]]
    mat_df_c = mat_df.iloc[correspond.T[1]]
    stats = pd.DataFrame(columns=['python', 'matlab'])
    for key in ['fishLength', 'fishRange', 'Err']:
        stats.loc[key, 'python'] = '{:6.2f} ± {:6.2f}'.format(py_df_c[key].mean(), py_df_c[key].std())
        stats.loc[key, 'matlab'] = '{:6.2f} ± {:6.2f}'.format(mat_df_c[key].mean(), mat_df_c[key].std())

    stats.loc['nTotal', 'python'] = '{}'.format(len(py_df_c))
    stats.loc['nTotal', 'matlab'] = '{}'.format(len(mat_df_c))
    print(tabulate(stats, headers='keys', tablefmt='psql', stralign='right'))

    print('Only QC > 0')
    is_qc = (mat_df_c['QC'] > 0).values
    mat_df_c = mat_df_c[is_qc]
    py_df_c = py_df_c[is_qc]
    stats = pd.DataFrame(columns=['python', 'matlab'])
    for key in ['fishLength', 'fishRange', 'Err']:
        stats.loc[key, 'python'] = '{:6.2f} ± {:6.2f}'.format(py_df_c[key].mean(), py_df_c[key].std())
        stats.loc[key, 'matlab'] = '{:6.2f} ± {:6.2f}'.format(mat_df_c[key].mean(), mat_df_c[key].std())

    stats.loc['nTotal', 'python'] = '{}'.format(len(py_df_c))
    stats.loc['nTotal', 'matlab'] = '{}'.format(len(mat_df_c))
    print(tabulate(stats, headers='keys', tablefmt='psql', stralign='right'))


def _read_kresimir_results():
    import scipy.io
    import pandas as pd
    mat = scipy.io.loadmat(expanduser('~/data/camtrawl_stereo_sample_data/Haul_83/Haul_083_qcresult.mat'))
    header = ub.readfrom(expanduser('~/data/camtrawl_stereo_sample_data/Haul_83/mat_file_header.csv')).strip().split(',')
    data = mat['lengthsqc']

    mat_df = pd.DataFrame(data, columns=header)
    mat_df['current_frame'] = mat_df['current_frame'].astype(np.int)
    mat_df['Species'] = mat_df['Species'].astype(np.int)
    mat_df['QC'] = mat_df['QC'].astype(np.int)

    # Transform so each row corresponds to one set of (x, y) points per detection
    bbox_cols1 = ['LX1', 'LX2', 'LX3', 'LX4', 'LY1', 'LY2', 'LY3', 'LY4', 'Lar', 'LboxL', 'WboxL', 'aveL']
    bbox_pts1 = mat_df[bbox_cols1[0:8]]  # NOQA
    bbox_pts1_ = bbox_pts1.values
    bbox_pts1_ = bbox_pts1_.reshape(len(bbox_pts1_), 2, 4).transpose((0, 2, 1))
    # mat_df['box_pts1'] = [pts for pts in bbox_pts1_]

    bbox_cols2 = ['RX1', 'RX2', 'RX3', 'RX4', 'RY1', 'RY2', 'RY3', 'RY4', 'Rar', 'LboxR', 'WboxR', 'aveW']
    bbox_pts2 = mat_df[bbox_cols2]  # NOQA
    bbox_pts2 = mat_df[bbox_cols2[0:8]]  # NOQA
    bbox_pts2_ = bbox_pts2.values
    bbox_pts2_ = bbox_pts2_.reshape(len(bbox_pts2_), 2, 4).transpose((0, 2, 1))
    # mat_df['box_pts2'] = [pts for pts in bbox_pts2_]

    mat_df['obox1'] = [ctalgo.OrientedBBox(*cv2.minAreaRect(pts[:, None, :].astype(np.int))) for pts in bbox_pts1_]
    mat_df['obox2'] = [ctalgo.OrientedBBox(*cv2.minAreaRect(pts[:, None, :].astype(np.int))) for pts in bbox_pts2_]

    mat_df.drop(bbox_cols2, axis=1, inplace=True)
    mat_df.drop(bbox_cols1, axis=1, inplace=True)

    return mat_df


if __name__ == '__main__':
    r"""
    CommandLine:

        workon_py2
        source ~/code/VIAME/build/install/setup_viame.sh
        export SPROKIT_PYTHON_MODULES=camtrawl_processes:kwiver.processes:viame.processes
        export PYTHONPATH=$(pwd):$PYTHONPATH

        python ~/code/VIAME/plugins/camtrawl/python/camtrawl_demo.py

        ffmpeg -y -f image2 -i out_haul83/%*.png -vcodec mpeg4 -vf "setpts=10*PTS" haul83-results.avi
    """
    import camtrawl_demo
    measurements = camtrawl_demo.demo()
