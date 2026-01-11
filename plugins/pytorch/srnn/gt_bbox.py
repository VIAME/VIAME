# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from kwiver.vital.types import BoundingBoxD
from enum import Enum

class GTFileType(Enum):
    MOT = 1
    AFRL = 2

class GTBBox(object):
    def __init__(self, gt_file_path, file_format):
        if file_format is GTFileType.MOT:
            print("*********MOT GT*******************")
            self._frame_track_dict = self._process_gt_MOT_file(gt_file_path) 
        elif file_format is GTFileType.AFRL:
            print("*********AFRL GT*******************")
            self._frame_track_dict = self._process_gt_AFRL_file(gt_file_path) 


    def _process_gt_MOT_file(self, gt_file_path):
        r"""Process MOT gt file
            The output of the function is a dictionary with following format
            [frame_num : [(id_num, bb_left, bb_top, bb_width, bb_height)]]
        """
        frame_track_dict = {}
        with open(gt_file_path, 'r') as f:
            for line in f:
                cur_line_list = line.rstrip('\n').split(',')
                frame_num, id_num = map(int, cur_line_list[:2])
                bb = tuple(map(float, cur_line_list[2:6]))
                frame_track_dict.setdefault(frame_num, []).append((id_num,) + bb)
    
        return frame_track_dict

    def _process_gt_AFRL_file(self, gt_file_path):
        r"""Process KW18 gt file
            The output of the function is a dictionary with following format
            [frame_num : [(id_num, bb_left, bb_top, bb_width, bb_height)]]
        """
        frame_track_dict = {}
        with open(gt_file_path, 'r') as f:
            for line in f:
                cur_line_list = line.rstrip('\n').split(' ')
                if cur_line_list[0][0] == '#':
                    continue

                frame_id = int(cur_line_list[2])
                track_id = int(cur_line_list[0])
                bbox_x = int(float(cur_line_list[9]))
                bbox_y = int(float(cur_line_list[10]))
                bbox_w = int(float(cur_line_list[11])) - bbox_x
                bbox_h = int(float(cur_line_list[12])) - bbox_y
    
                bb = bbox_x, bbox_y, bbox_w, bbox_h 
                frame_track_dict.setdefault(frame_id, []).append((track_id,) + bb)
        
        return frame_track_dict

    def __getitem__(self, f_id):
        try:
            bb_info = self._frame_track_dict[f_id]
        except KeyError:
            print('frame id: {} does not exist!'.format(f_id))
            exit(0)
        
        bb_list = []
        
        for item in bb_info:
            x, y, w, h = map(float, item[1:])
            bb_list.append(BoundingBoxD(x, y, x + w, y + h))

        return bb_list

