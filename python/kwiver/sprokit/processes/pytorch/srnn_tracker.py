# ckwg +29
# Copyright 2018-2019 by Kitware, Inc.
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

import sys
import torch

from torchvision import models, transforms
from torch.autograd import Variable

import numpy as np
import scipy as sp
import scipy.optimize

from timeit import default_timer as timer
from PIL import Image as pilImage

from kwiver.sprokit.pipeline import process
from kwiver.sprokit.processes.kwiver_process import KwiverProcess

from kwiver.vital.types import Image
from kwiver.vital.types import DetectedObject, DetectedObjectSet
from kwiver.vital.types import ObjectTrackState, Track, ObjectTrackSet
from kwiver.vital.types import new_descriptor

from .utils.track import track_state, track, track_set
from kwiver.vital.util.VitalPIL import get_pil_image

from .utils.models import Siamese
from .utils.grid import Grid
from .utils.srnn_matching import SRNNMatching, RnnType
from .utils.siamese_feature_extractor import SiameseFeatureExtractor
from .utils.iou_tracker import IOUTracker
from .utils.parse_gpu_list import gpu_list_desc, parse_gpu_list
from .utils.gt_bbox import GTBBox, GTFileType
from .utils.models import get_config

g_config = get_config()

def ts2ot_list(track_set):
    ot_list = [Track(id=t.track_id) for t in track_set]

    for idx, t in enumerate(track_set):
        ot = ot_list[idx]
        for ti in t:
            ot_state = ObjectTrackState(ti.sys_frame_id, ti.sys_frame_time,
                                        ti.detected_object)
            if not ot.append(ot_state):
                print('Error: Cannot add ObjectTrackState')
    return ot_list

class SRNNTracker(KwiverProcess):
    # ----------------------------------------------
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        self.__declare_config_traits()

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
        self.declare_input_port_using_trait('timestamp', required)
        self.declare_input_port_using_trait('object_track_set', optional)

        #  output port ( port-name,flags)
        self.declare_output_port_using_trait('object_track_set', optional)
        self.declare_output_port_using_trait('detected_object_set', optional)

    def __declare_config_traits(self):
        def add_declare_config_trait(name, default, desc):
            self.add_config_trait(name, name, default, desc)
            self.declare_config_using_trait(name)

        #GPU list
        add_declare_config_trait('gpu_list', 'all',
                                 gpu_list_desc(use_for='SRNN tracking'))

        # siamese
        #----------------------------------------------------------------------------------
        add_declare_config_trait('siamese_model_path',
                                 'siamese/snapshot_epoch_6.pt',
                                 'Trained PyTorch model.')

        add_declare_config_trait('siamese_model_input_size', '224',
                                 'Model input image size')

        add_declare_config_trait('siamese_batch_size', '128',
                                 'siamese model processing batch size')
        #----------------------------------------------------------------------------------

        # detection select threshold
        add_declare_config_trait('detection_select_threshold', '0.0',
                                 'detection select threshold')
        add_declare_config_trait('track_initialization_threshold', '0.0',
                                 'track initialization threshold')

        # SRNN
        #----------------------------------------------------------------------------------
        # target RNN full model
        add_declare_config_trait("targetRNN_AIM_model_path",
                                 'targetRNN_snapshot/App_LSTM_epoch_51.pt',
                                 'Trained targetRNN PyTorch model.')

        # target RNN AI model
        add_declare_config_trait("targetRNN_AIM_V_model_path",
                                 'targetRNN_AI/App_LSTM_epoch_51.pt',
                                 'Trained targetRNN AIM with variable input size PyTorch model.')

        # target RNN batch size
        add_declare_config_trait("targetRNN_batch_size", '256',
                                 'targetRNN model processing batch size')

        # matching similarity threshold
        add_declare_config_trait("similarity_threshold", '0.5',
                                 'similarity threshold.')
        #----------------------------------------------------------------------------------

        # IOU
        #----------------------------------------------------------------------------------
        # IOU tracker flag
        add_declare_config_trait("IOU_tracker_flag", 'True', 'IOU tracker flag.')

        # IOU accept threshold
        add_declare_config_trait("IOU_accept_threshold", '0.5',
                                 'IOU accept threshold.')

        # IOU reject threshold
        add_declare_config_trait("IOU_reject_threshold", '0.1',
                                 'IOU reject threshold.')
        #----------------------------------------------------------------------------------

        # search threshold
        add_declare_config_trait("track_search_threshold", '0.1',
                                 'track search threshold.')

        # matching active track threshold
        add_declare_config_trait("terminate_track_threshold", '15',
                                 'terminate the tracking if the target has been lost for more than '
                                 'terminate_track_threshold read-in frames.')

        # matching active track threshold
        add_declare_config_trait("sys_terminate_track_threshold", '50',
                                 'terminate the tracking if the target has been lost for more than '
                                 'terminate_track_threshold system (original) frames.')

        # MOT gt detection
        #-------------------------------------------------------------------
        add_declare_config_trait("MOT_GTbbox_flag", 'False', 'MOT GT bbox flag')
        #-------------------------------------------------------------------

        # AFRL gt detection
        #-------------------------------------------------------------------
        add_declare_config_trait("AFRL_GTbbox_flag", 'False', 'AFRL GT bbox flag')
        #-------------------------------------------------------------------

        # GT bbox file
        #-------------------------------------------------------------------
        add_declare_config_trait("GT_bbox_file_path", '',
                                 'ground truth detection file for testing')
        #-------------------------------------------------------------------

        # Add features to detections
        #-------------------------------------------------------------------
        add_declare_config_trait("add_features_to_detections", 'True',
                                 'Should we add internally computed features to detections?')
        #-------------------------------------------------------------------

    # ----------------------------------------------
    def _configure(self):
        self._select_threshold = float(self.config_value('detection_select_threshold'))
        self._track_initialization_threshold = float(self.config_value('track_initialization_threshold'))

        #GPU_list
        self._gpu_list = parse_gpu_list(self.config_value('gpu_list'))

        # Siamese model config
        siamese_img_size = int(self.config_value('siamese_model_input_size'))
        siamese_batch_size = int(self.config_value('siamese_batch_size'))
        siamese_model_path = self.config_value('siamese_model_path')
        self._app_feature_extractor = SiameseFeatureExtractor(siamese_model_path,
                siamese_img_size, siamese_batch_size, self._gpu_list)

        # targetRNN_full model config
        targetRNN_batch_size = int(self.config_value('targetRNN_batch_size'))
        targetRNN_AIM_model_path = self.config_value('targetRNN_AIM_model_path')
        targetRNN_AIM_V_model_path = self.config_value('targetRNN_AIM_V_model_path')
        self._srnn_matching = SRNNMatching(targetRNN_AIM_model_path,
                targetRNN_AIM_V_model_path, targetRNN_batch_size, self._gpu_list)

        self._gtbbox_flag = False
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

        self._gtbbox_flag = MOT_GT_flag or AFRL_GT_flag

        # read GT bbox related
        if self._gtbbox_flag:
            gtbbox_file_path = self.config_value('GT_bbox_file_path')
            self._m_bbox = GTBBox(gtbbox_file_path, file_format)

        self._similarity_threshold = float(self.config_value('similarity_threshold'))

        # IOU tracker
        iou_accept_threshold = float(self.config_value('IOU_accept_threshold'))
        iou_reject_threshold = float(self.config_value('IOU_reject_threshold'))
        self._iou_tracker = IOUTracker(iou_accept_threshold, iou_reject_threshold)

        # track search threshold
        self._ts_threshold = float(self.config_value('track_search_threshold'))
        self._grid = Grid()
        # generated track_set
        self._track_set = track_set()
        self._terminate_track_threshold = int(self.config_value('terminate_track_threshold'))
        self._sys_terminate_track_threshold = int(self.config_value('sys_terminate_track_threshold'))
        # add features to detections?
        self._add_features_to_detections = \
                (self.config_value('add_features_to_detections') == 'True')
        self._base_configure()

    # ----------------------------------------------
    def _step(self):
        try:
            def timing(desc, f):
                """Return f(), printing a message about how long it took"""
                start = timer()
                result = f()
                end = timer()
                print('%%%', desc, ' elapsed time: ', end - start, sep='')
                return result

            print('step', self._step_id)

            # grab image container from port using traits
            in_img_c = self.grab_input_using_trait('image')
            timestamp = self.grab_input_using_trait('timestamp')
            dos_ptr = self.grab_input_using_trait('detected_object_set')
            print('timestamp =', repr(timestamp))

            # Get current frame
            im = get_pil_image(in_img_c.image()).convert('RGB')

            # Get detection bbox
            if self._gtbbox_flag:
                dos = self._m_bbox[self._step_id]
                bbox_num = len(dos)
            else:
                dos = dos_ptr.select(self._select_threshold)
                bbox_num = dos.size()
            #print('bbox list len is', dos.size())

            det_obj_set = DetectedObjectSet()
            if bbox_num == 0:
                print('!!! No bbox is provided on this frame.  Skipping this frame !!!')
            else:
                # interaction features
                grid_feature_list = timing('grid feature', lambda:
                                           self._grid(im.size, dos, self._gtbbox_flag))

                # appearance features (format: pytorch tensor)
                pt_app_features = timing('app feature', lambda:
                                         self._app_feature_extractor(im, dos, self._gtbbox_flag))

                track_state_list = []
                next_track_id = int(self._track_set.get_max_track_id()) + 1

                # get new track state from new frame and detections
                for idx, item in enumerate(dos):
                    if self._gtbbox_flag:
                        bbox = item
                        fid = self._step_id
                        ts = self._step_id
                        d_obj = DetectedObject(bbox=item, confidence=1.0)
                    else:
                        bbox = item.bounding_box()
                        fid = timestamp.get_frame()
                        ts = timestamp.get_time_usec()
                        d_obj = item

                    if self._add_features_to_detections:
                        # store app feature to detected_object
                        app_f = new_descriptor(g_config.A_F_num)
                        app_f[:] = pt_app_features[idx].numpy()
                        d_obj.set_descriptor(app_f)
                    det_obj_set.add(d_obj)

                    # build track state for current bbox for matching
                    cur_ts = track_state(frame_id=self._step_id,
                                        bbox_center=bbox.center(),
                                        interaction_feature=grid_feature_list[idx],
                                        app_feature=pt_app_features[idx],
                                        bbox=[int(bbox.min_x()), int(bbox.min_y()),
                                              int(bbox.width()), int(bbox.height())],
                                        detected_object=d_obj,
                                        sys_frame_id=fid, sys_frame_time=ts)
                    track_state_list.append(cur_ts)

                # if there are no tracks, generate new tracks from the track_state_list
                if not self._track_flag:
                    next_track_id = self._track_set.add_new_track_state_list(next_track_id,
                                    track_state_list, self._track_initialization_threshold)
                    self._track_flag = True
                else:
                    # check whether we need to terminate a track
                    for track in list(self._track_set.iter_active()):
                        # terminating a track based on readin_frame_id or original_frame_id gap
                        if (self._step_id - track[-1].frame_id > self._terminate_track_threshold
                            or fid - track[-1].sys_frame_id > self._sys_terminate_track_threshold):
                            self._track_set.deactivate_track(track)


                    # call IOU tracker
                    if self._IOU_flag:
                        self._track_set, track_state_list = timing('IOU tracking', lambda: (
                            self._iou_tracker(self._track_set, track_state_list)
                        ))

                    #print('***track_set len', len(self._track_set))
                    #print('***track_state_list len', len(track_state_list))

                    # estimate similarity matrix
                    similarity_mat, track_idx_list = timing('SRNN association', lambda: (
                        self._srnn_matching(self._track_set, track_state_list, self._ts_threshold)
                    ))

                    # reset update_flag
                    self._track_set.reset_updated_flag()

                    # Hungarian algorithm
                    row_idx_list, col_idx_list = timing('Hungarian algorithm', lambda: (
                        sp.optimize.linear_sum_assignment(similarity_mat)
                    ))

                    for i in range(len(row_idx_list)):
                        r = row_idx_list[i]
                        c = col_idx_list[i]

                        if -similarity_mat[r, c] < self._similarity_threshold:
                            # initialize a new track
                            if (track_state_list[c].detected_object.confidence()
                                   >= self._track_initialization_threshold):
                                self._track_set.add_new_track_state(next_track_id,
                                        track_state_list[c])
                                next_track_id += 1
                        else:
                            # add to existing track
                            self._track_set.update_track(track_idx_list[r], track_state_list[c])

                    # for the remaining unmatched track states, we initialize new tracks
                    if len(track_state_list) - len(col_idx_list) > 0:
                        for i in range(len(track_state_list)):
                            if (i not in col_idx_list
                                and (track_state_list[i].detected_object.confidence()
                                     >= self._track_initialization_threshold)):
                                self._track_set.add_new_track_state(next_track_id,
                                        track_state_list[i])
                                next_track_id += 1

                print('total tracks', len(self._track_set))

            # push track set to output port
            ot_list = ts2ot_list(self._track_set)
            ots = ObjectTrackSet(ot_list)

            self.push_to_port_using_trait('object_track_set', ots)
            self.push_to_port_using_trait('detected_object_set', det_obj_set)

            self._step_id += 1

            self._base_step()

        except BaseException as e:
            print( repr( e ) )
            import traceback
            print( traceback.format_exc() )
            sys.stdout.flush()
            raise


# ==================================================================
def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory

    module_name = 'python:kwiver.pytorch.SRNNTracker'

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process('srnn_tracker',
                                'Structural RNN based tracking',
                                SRNNTracker)

    process_factory.mark_process_module_as_loaded(module_name)
