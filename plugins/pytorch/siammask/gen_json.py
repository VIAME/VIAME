from os.path import join, basename, dirname, exists
import json
from glob import glob


def gen_json(image_folder, save_folder):
    def get_im_num(line, adj=0):
        return int(line[2])

    def get_bbox(line):
        return [int(float(i)) for i in line[3:7]]

    dataDir = '.'
    dataset = dict()
    csvs = sorted(glob(join(image_folder, '*', '*.csv')))
    for annFile in csvs:
        with open(annFile) as f:
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
        video_crop_base_path = basename(dirname(annFile))
        dataset[video_crop_base_path] = dict()
        for track_idx, track in enumerate(track_nums):
            for line_idx, line in enumerate(tracks[track]):
                im_num = get_im_num(line, min_im)
                bbox = get_bbox(line)
                z_path = join(save_folder, "crop511", video_crop_base_path, f'{im_num:08}.{track_idx:08}.z.jpg')
                x_path = join(save_folder, "crop511", video_crop_base_path, f'{im_num:08}.{track_idx:08}.x.jpg')
                if not exists(z_path) or not exists(x_path):
                    continue
                if line_idx == 0 or f'{track_idx:08}' not in dataset[video_crop_base_path]:
                    dataset[video_crop_base_path][f'{track_idx:08}'] = {f'{im_num:08}': bbox}
                else:
                    dataset[video_crop_base_path][f'{track_idx:08}'][f'{im_num:08}'] = bbox

    print('save json (dataset), please wait 20 seconds~')
    json.dump(dataset, open(join(save_folder, 'dataset.json'), 'w'), indent=4, sort_keys=True)
    print('done!')
