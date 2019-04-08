# ckwg +28
# Copyright 2018 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from vital.types import BoundingBox
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
            bb_list.append(BoundingBox(x, y, x + w, y + h))

        return bb_list

