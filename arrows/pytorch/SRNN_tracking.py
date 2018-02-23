# ckwg +28
# Copyright 2017 by Kitware, Inc.
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

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch import nn
import numpy as np
import scipy as sp
import scipy.optimize

from PIL import Image as pilImage

from sprokit.pipeline import process
from kwiver.kwiver_process import KwiverProcess
from vital.types import Image
from vital.types import DetectedObject, DetectedObjectSet
from vital.types import new_descriptor

from timeit import default_timer as timer

from vital.types import ( 
    ObjectTrackState,
    Track,
    ObjectTrackSet
)

from vital.util.VitalPIL import get_pil_image

from kwiver.arrows.pytorch.models import Siamese
from kwiver.arrows.pytorch.grid import grid
from kwiver.arrows.pytorch.track import track_state, track, track_set
from kwiver.arrows.pytorch.SRNN_matching import SRNN_matching, RnnType
from kwiver.arrows.pytorch.pytorch_siamese_f_extractor import pytorch_siamese_f_extractor
from kwiver.arrows.pytorch.iou_tracking import IOU_tracker

from kwiver.arrows.pytorch.MOT_bbox import MOT_bbox, GTFileType
from kwiver.arrows.pytorch.models import get_config

g_config = get_config()

def ts2ot_list(track_set):
    ot_list = [] 
    for t in track_set:
        ot = Track(id=t.id)
        ot_list.append(ot)

    for idx, t in enumerate(track_set):
        for i in range(len(t)):
            ot_state = ObjectTrackState(t[i].frame_id, t[i].detectedObj)
            if not ot_list[idx].append(ot_state):
                print('cannot add ObjectTrackState')
                exit(1)

    return ot_list


class SRNN_tracking(KwiverProcess):

    # ----------------------------------------------
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)
        #GPU list
        self.add_config_trait("GPU_list", "GPU_list", 'all',
                              'define which GPU to use for SRNN tracking. e.g., all, 1,2')
        self.declare_config_using_trait('GPU_list')

        # siamese 
        #----------------------------------------------------------------------------------
        self.add_config_trait("siamese_model_path", "siamese_model_path",
                              '/home/bdong/HiDive_project/tracking_the_untrackable/snapshot/siamese/snapshot_epoch_6.pt',
                              'Trained PyTorch model.')
        self.declare_config_using_trait('siamese_model_path')

        self.add_config_trait("siamese_model_input_size", "siamese_model_input_size", '224',
                              'Model input image size')
        self.declare_config_using_trait('siamese_model_input_size')

        self.add_config_trait("siamese_batch_size", "siamese_batch_size", '128',
                              'siamese model processing batch size')
        self.declare_config_using_trait('siamese_batch_size')
        #----------------------------------------------------------------------------------

        # detection select threshold
        self.add_config_trait("detection_select_threshold", "detection_select_threshold", '0.0',
                              'detection select threshold')
        self.declare_config_using_trait('detection_select_threshold')

        # SRNN
        #----------------------------------------------------------------------------------
        # target RNN full model
        self.add_config_trait("targetRNN_AIM_model_path", "targetRNN_AIM_model_path",
                              '/home/bdong/HiDive_project/tracking_the_untrackable/snapshot/targetRNN_snapshot/App_LSTM_epoch_51.pt',
                              'Trained targetRNN PyTorch model.')
        self.declare_config_using_trait('targetRNN_AIM_model_path')

        # target RNN AI model
        self.add_config_trait("targetRNN_AIM_V_model_path", "targetRNN_AIM_V_model_path",
                              '/home/bdong/HiDive_project/tracking_the_untrackable/snapshot/targetRNN_AI/App_LSTM_epoch_51.pt',
                              'Trained targetRNN AIM with variable input size PyTorch model.')
        self.declare_config_using_trait('targetRNN_AIM_V_model_path')

        # target RNN batch size
        self.add_config_trait("targetRNN_batch_size", "targetRNN_batch_size", '256',
                              'targetRNN model processing batch size')
        self.declare_config_using_trait('targetRNN_batch_size')

        # matching similarity threshold
        self.add_config_trait("similarity_threshold", "similarity_threshold", '0.5',
                              'similarity threshold.')
        self.declare_config_using_trait('similarity_threshold')
        #----------------------------------------------------------------------------------

        # IOU
        #----------------------------------------------------------------------------------
        # IOU tracker flag
        self.add_config_trait("IOU_tracker_flag", "IOU_tracker_flag", 'True', 'IOU tracker flag.')
        self.declare_config_using_trait('IOU_tracker_flag')

        # IOU accept threshold
        self.add_config_trait("IOU_accept_threshold", "IOU_accept_threshold", '0.5',
                              'IOU accept threshold.')
        self.declare_config_using_trait('IOU_accept_threshold')
        
        # IOU reject threshold
        self.add_config_trait("IOU_reject_threshold", "IOU_reject_threshold", '0.1',
                              'IOU reject threshold.')
        self.declare_config_using_trait('IOU_reject_threshold')
        #----------------------------------------------------------------------------------
        
        # search threshold
        self.add_config_trait("track_search_threshold", "track_search_threshold", '0.1',
                              'track search threshold.')
        self.declare_config_using_trait('track_search_threshold')

        # matching active track threshold
        self.add_config_trait("terminate_track_threshold", "terminate_track_threshold", '15',
                              'terminate the tracking if the target has been lost for more than termniate_track_threshold frames.')
        self.declare_config_using_trait('terminate_track_threshold')

        # MOT gt detection
        #-------------------------------------------------------------------
        self.add_config_trait("MOT_GTbbox_flag", "MOT_GTbbox_flag", 'False', 'MOT GT bbox flag')
        self.declare_config_using_trait('MOT_GTbbox_flag')
        #-------------------------------------------------------------------

        # AFRL gt detection
        #-------------------------------------------------------------------
        self.add_config_trait("AFRL_GTbbox_flag", "AFRL_GTbbox_flag", 'False', 'AFRL GT bbox flag')
        self.declare_config_using_trait('AFRL_GTbbox_flag')
        
        #-------------------------------------------------------------------

        # GT bbox file
        #-------------------------------------------------------------------
        self.add_config_trait("GT_bbox_file_path", "GT_bbox_file_path", 
                             '', 'ground truth detection file for testing')
        self.declare_config_using_trait('GT_bbox_file_path')
        #-------------------------------------------------------------------

        self._track_flag = False

        # AFRL start id : 0
        # MOT start id : 1
        self._step_id = 0

        # set up required flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        #  input port ( port-name,flags)
        # self.declare_input_port_using_trait('framestamp', optional)
        self.declare_input_port_using_trait('image', required)
        self.declare_input_port_using_trait('detected_object_set', required)
        self.declare_input_port_using_trait('object_track_set', optional)

        #  output port ( port-name,flags)
        self.declare_output_port_using_trait('object_track_set', optional)
        self.declare_output_port_using_trait('detected_object_set', optional)

    # ----------------------------------------------
    def _configure(self):
        self._select_threshold = float(self.config_value('detection_select_threshold'))

        #GPU_list
        GPU_list_str = self.config_value('GPU_list')
        if GPU_list_str == 'all':
            self._GPU_list = None
        else:
            self._GPU_list = list(map(int, GPU_list_str.split(',')))

        # Siamese model config
        siamese_img_size = int(self.config_value('siamese_model_input_size'))
        siamese_batch_size = int(self.config_value('siamese_batch_size'))
        siamese_model_path = self.config_value('siamese_model_path')
        self._app_feature_extractor = pytorch_siamese_f_extractor(siamese_model_path, siamese_img_size, siamese_batch_size, self._GPU_list)

        # targetRNN_full model config
        targetRNN_batch_size = int(self.config_value('targetRNN_batch_size'))
        targetRNN_AIM_model_path = self.config_value('targetRNN_AIM_model_path')
        targetRNN_AIM_V_model_path = self.config_value('targetRNN_AIM_V_model_path')
        self._SRNN_matching = SRNN_matching(targetRNN_AIM_model_path, targetRNN_AIM_V_model_path, targetRNN_batch_size, self._GPU_list)

        self._GTbbox_flag = False
        # use MOT gt detection
        MOT_GTbbox_flag = self.config_value('MOT_GTbbox_flag')
        MOT_GT_flag = (MOT_GTbbox_flag == 'True')
        if MOT_GT_flag:
            file_format = GTFileType.MOT

        # use AFRL gt detection
        AFRL_GTbbox_flag = self.config_value('AFRL_GTbbox_flag')
        AFRL_GT_flag = (AFRL_GTbbox_flag == 'True')
        if AFRL_GT_flag:
            file_format = GTFileType.AFRL

        # IOU tracker flag
        self._IOU_flag = True
        IOU_flag = self.config_value('IOU_tracker_flag')
        self._IOU_flag = (IOU_flag == 'True')

        self._GTbbox_flag = MOT_GT_flag or AFRL_GT_flag 

        # read GT bbox related
        if self._GTbbox_flag: 
            GTbbox_file_path = self.config_value('GT_bbox_file_path')
            self._m_bbox = MOT_bbox(GTbbox_file_path, file_format)

        self._similarity_threshold = float(self.config_value('similarity_threshold'))

        # IOU tracker
        iou_accept_threshold = float(self.config_value('IOU_accept_threshold'))
        iou_reject_threshold = float(self.config_value('IOU_reject_threshold'))
        self._iou_tracker = IOU_tracker(iou_accept_threshold, iou_reject_threshold)

        # track search threshold
        self._ts_threshold = float(self.config_value('track_search_threshold'))

        self._grid = grid()

        # generated track_set
        self._track_set = track_set()
        self._terminate_track_threshold = int(self.config_value('terminate_track_threshold'))

        self._base_configure()

    # ----------------------------------------------
    def _step(self):
        print('step {}'.format(self._step_id))

        # grab image container from port using traits
        in_img_c = self.grab_input_using_trait('image')
        dos_ptr = self.grab_input_using_trait('detected_object_set')

        # Get current frame and give it to app feature extractor
        im = get_pil_image(in_img_c.image())
        self._app_feature_extractor.frame = im

        bbox_num = 0
        # Get detection bbox
        if self._GTbbox_flag is True:
            dos = self._m_bbox[self._step_id] 
            bbox_num = len(dos)
        else:
            dos = dos_ptr.select(self._select_threshold)
            bbox_num = dos.size()
        #print('bbox list len is {}'.format(dos.size()))

        det_obj_set = DetectedObjectSet()
        if bbox_num == 0:
            print('!!! No bbox is provided on this frame and skip this frame !!!')
        else:
            # interaction features
            grid_f_begin = timer()
            grid_feature_list = self._grid(im.size, dos, self._GTbbox_flag)
            grid_f_end = timer()
            print('%%%grid feature eclapsed time: {}'.format(grid_f_end - grid_f_begin))
            
            # appearance features (format: pytorch tensor)
            app_f_begin = timer()
            pt_app_features = self._app_feature_extractor(dos, self._GTbbox_flag) 
            app_f_end = timer()
            print('%%%app feature eclapsed time: {}'.format(app_f_end - app_f_begin))

            track_state_list = []
            next_trackID = int(self._track_set.get_max_track_ID()) + 1
        
            # get new track state from new frame and detections
            for idx, item in enumerate(dos):
                if self._GTbbox_flag is True:
                    bbox = item
                    d_obj = DetectedObject(bbox=item , confidence=1.0)
                else:
                    bbox = item.bounding_box()
                    d_obj = item

                # store app feature to detectedObject
                app_f = new_descriptor(g_config.A_F_num)
                app_f[:] = pt_app_features[idx].numpy()
                d_obj.set_descriptor(app_f)
                det_obj_set.add(d_obj)

                # build track state for current bbox for matching
                cur_ts = track_state(frame_id=self._step_id, bbox_center=tuple((bbox.center())), 
                                     interaction_feature=grid_feature_list[idx],
                                     app_feature=pt_app_features[idx], bbox=[int(bbox.min_x()), int(bbox.min_y()), 
                                                                    int(bbox.width()), int(bbox.height())],
                                     detectedObject=d_obj)
                track_state_list.append(cur_ts)
                
            # if there is no tracks, generate new tracks from the track_state_list
            if self._track_flag is False:
                next_trackID = self._track_set.add_new_track_state_list(next_trackID, track_state_list)
                self._track_flag = True
            else:
                # check whether we need to terminate a track
                for ti in range(len(self._track_set)):
                    if self._step_id - self._track_set[ti][-1].frame_id > self._terminate_track_threshold:
                        self._track_set[ti].active_flag = False

                #print('track_set len {}'.format(len(self._track_set)))
                #print('track_state_list len {}'.format(len(track_state_list)))
                
                # call IOU tracker
                if self._IOU_flag is True:
                    IOU_begin = timer()
                    self._track_set, track_state_list = self._iou_tracker(self._track_set, track_state_list)
                    IOU_end = timer()
                    print('%%%IOU tracking eclapsed time: {}'.format(IOU_end - IOU_begin))

                #print('***track_set len {}'.format(len(self._track_set)))
                #print('***track_state_list len {}'.format(len(track_state_list)))

                # estimate similarity matrix
                ttu_begin = timer()
                similarity_mat, track_idx_list = self._SRNN_matching(self._track_set, track_state_list, self._ts_threshold)
                ttu_end = timer()
                print('%%%SRNN assication eclapsed time: {}'.format(ttu_end - ttu_begin))

                #reset update_flag
                self._track_set.reset_updated_flag()

                # Hungarian algorithm
                hung_begin = timer()
                row_idx_list, col_idx_list = sp.optimize.linear_sum_assignment(similarity_mat)
                hung_end = timer()
                print('%%%Hungarian alogrithm eclapsed time: {}'.format(hung_end - hung_begin))
                
                for i in range(len(row_idx_list)):
                    r = row_idx_list[i]
                    c = col_idx_list[i]

                    if -similarity_mat[r, c] < self._similarity_threshold:
                        # initialize a new track
                        self._track_set.add_new_track_state(next_trackID, track_state_list[c])
                        next_trackID += 1
                    else:
                        # add to existing track
                        self._track_set.update_track(track_idx_list[r], track_state_list[c])
                
                # for rest unmatched track_state, we initialize new tracks
                if len(track_state_list) - len(col_idx_list) > 0:
                    for i in range(len(track_state_list)):
                        if i not in col_idx_list:
                            self._track_set.add_new_track_state(next_trackID, track_state_list[i])
                            next_trackID += 1

            print('total tracks {}'.format(len(self._track_set)))

        # push track set to output port
        ot_list = ts2ot_list(self._track_set)
        ots = ObjectTrackSet(ot_list)

        self.push_to_port_using_trait('object_track_set', ots)
        self.push_to_port_using_trait('detected_object_set', det_obj_set)

        self._step_id += 1

        self._base_step()

    def __del__(self):
        print('!!!!SRNN tracking Deleting python process!!!!')

# ==================================================================
def __sprokit_register__():
    from sprokit.pipeline import process_factory

    module_name = 'python:kwiver.SRNN_tracking'

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process('SRNN_tracking', 'Structural RNN based tracking',
                                SRNN_tracking)

    process_factory.mark_process_module_as_loaded(module_name)

