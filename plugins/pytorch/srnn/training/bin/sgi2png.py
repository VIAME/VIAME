# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import argparse
import glob
import os
import os.path as osp

from PIL import Image
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='processing kw18 file for generating training data')
    parser.add_argument('--root_path', type=str, dest='root_path',
                        help='The root path contains all sgi image',
                        default='/home/bdong/NOAA/v1_annotations')

    parser.add_argument('--out_path', type=str, dest='out_path',
                        help='The output path for storing all png images',
                        default='/home/bdong/NOAA/v1_annotations_PNG')

    args = parser.parse_args()

    in_img_list = glob.glob(osp.join(args.root_path, '**', '*.sgi'), recursive=True)

    pbar = tqdm(total=len(in_img_list), desc='Processing SGI image...')

    for img_name in in_img_list:
        cur_path_list = img_name.rstrip('\n').split('/')
        png_name = cur_path_list[-1][:-3] + 'png'
        new_image_path = osp.join(args.out_path, cur_path_list[-2])
        os.makedirs(new_image_path, exist_ok=True)

        png_path = osp.join(new_image_path, png_name)

        if osp.exists(png_path):
            pbar.update(1)
            continue
        else:
            Image.open(img_name).save(png_path)
            pbar.update(1)
