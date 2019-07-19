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

from sprokit.pipeline import process
from kwiver.kwiver_process import KwiverProcess

from vital.types import Image
from vital.types import DetectedObject, DetectedObjectSet
from vital.types import ObjectTrackState, Track, ObjectTrackSet
from vital.types import new_descriptor
from vital.types import BoundingBox

from kwiver.arrows.pytorch.track import track_state, track, track_set
from vital.util.VitalPIL import get_pil_image

from kwiver.arrows.pytorch.models import Siamese
from kwiver.arrows.pytorch.grid import Grid
from kwiver.arrows.pytorch.srnn_matching import SRNNMatching, RnnType
from kwiver.arrows.pytorch.siamese_feature_extractor import SiameseFeatureExtractor
from kwiver.arrows.pytorch.iou_tracker import IOUTracker
from kwiver.arrows.pytorch.parse_gpu_list import gpu_list_desc, parse_gpu_list
from kwiver.arrows.pytorch.gt_bbox import GTBBox, GTFileType
from kwiver.arrows.pytorch.models import get_config

# from kwiver.arrows.pytorch.pysot.tools.kwiver_test import pysot_step

import ast
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain

def pysot_step(img, idx, gt_bbox, the_model, the_config, tracker=None):
    cfg.merge_from_file(the_config)
    # build tracker
    pred_bboxes = []
    scores = []
    #if idx == 0:
    if tracker is None:
        model = ModelBuilder()
        model = load_pretrain(model, the_model).cuda().eval()
        tracker = build_tracker(model)

        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
        gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
        tracker.init(img, gt_bbox_)
        pred_bbox = gt_bbox_
        scores.append(None)
        pred_bboxes.append(pred_bbox)
        best_score = 1.0
    else:
        outputs = tracker.track(img)
        pred_bbox = outputs['bbox']
        pred_bboxes.append(pred_bbox)
        scores.append(outputs['best_score'])
        best_score = outputs['best_score']
    gt_bbox = list(map(int, gt_bbox))
    pred_bbox = list(map(int, pred_bbox))
    return pred_bbox, tracker, best_score


def st2ot_list(track_set):
    ot_list = [Track(id=t.track_id) for t in track_set]
    for idx, t in enumerate(track_set):
        ot = ot_list[idx]
        for ti in t:
            ot_state = ObjectTrackState(ti.sys_frame_id, ti.sys_frame_time,
                                        ti.detected_object)
            if not ot.append(ot_state):
                print('Error: Cannot add ObjectTrackState')
    return ot_list


class PYSOTTracker(KwiverProcess):
    # ----------------------------------------------
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        #GPU list
        #----------------------------------------------------------------------
        self.add_config_trait("gpu_list", "gpu_list", 'all',
                              gpu_list_desc(use_for='Siamese PYSOT tracking'))
        self.declare_config_using_trait('gpu_list')
        #----------------------------------------------------------------------

        # Config file
        #---------------------------------------------------------------
        self.add_config_trait("pysot_config_path", "pysot_config_path",
                              'models/pysot_config.yaml',
                              'PYSOT config file.')
        self.declare_config_using_trait("pysot_config_path")
        #---------------------------------------------------------------

        # Model file
        #-------------------------------------------------------------
        self.add_config_trait("pysot_model_path", "pysot_model_path",
                              'models/pysot_model.pth',
                              'Trained PYSOT model file.')
        self.declare_config_using_trait("pysot_model_path")
        #-------------------------------------------------------------

        self.add_config_trait("seed_track", "seed_track",
                              '[100, 100, 100, 100]',
                              'Location of starting GT bounding box.')
        self.declare_config_using_trait("seed_track")
        
        self.add_config_trait("track_th", "track_th",
                              '0.0',
                              'Minimum confidence to keep track.')
        self.declare_config_using_trait("track_th")        

        self.add_config_trait("ots_input_flag", "ots_input_flag",
                              'False',
                              'Flag determining if object_track_set is an input.')
        self.declare_config_using_trait("ots_input_flag")   

        # # PYSOT Tracker (based on siamrpn_r50_l234_dwxcorr/config.yaml)
        # #--------------------------------------------------------------------------------------
        # self.add_config_trait("pysot_model_exemplar_size", "pysot_model_exemplar_size", '127',
        #                       'Model exemplar image size')
        # self.declare_config_using_trait('pysot_model_exemplar_size')
        #
        # self.add_config_trait("pysot_model_instance_size", "pysot_model_instance_size", '255',
        #                       'Model input instance size')
        # self.declare_config_using_trait('pysot_model_instance_size')
        #
        # self.add_config_trait("pysot_batch_size", "pysot_batch_size", '28',
        #                       'pysot model processing batch size')
        # self.declare_config_using_trait('pysot_batch_size')
        # #--------------------------------------------------------------------------------------

        self.tracker = None
        self._track_flag = False

        self._step_id = 0

        # set up required flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        #  input port ( port-name,flags)
        self.declare_input_port_using_trait('image', required)
        self.declare_input_port_using_trait('timestamp', required)
        self.declare_input_port_using_trait('object_track_set', optional)
        #self.declare_input_port_using_trait('detected_object_set', optional)

        #  output port ( port-name,flags)
        self.declare_output_port_using_trait('object_track_set', optional)
        self.declare_output_port_using_trait('detected_object_set', optional)

        self._track_set = track_set()
        
        self._track_flag = False
        self._init_track_flag = False
        self._gt_bbox_flag = False
        self._first_frame_found_flag = False

    # ----------------------------------------------
    def _configure(self):
        #GPU_list
        self._gpu_list = parse_gpu_list(self.config_value('gpu_list'))

        self._model_path = self.config_value('pysot_model_path')
        self._config_path = self.config_value('pysot_config_path')
        self._track_th = float(self.config_value('track_th'))
        self._seed_bbox = ast.literal_eval(self.config_value("seed_track"))
        self._ots_input_flag = (self.config_value('ots_input_flag') == 'True')

        self._base_configure()

    # ----------------------------------------------
    def _step(self):
        try:
            print('step {}'.format(self._step_id))

            in_img_c = self.grab_input_using_trait('image')
            timestamp = self.grab_input_using_trait('timestamp')
            if self._ots_input_flag:
                input_ots = self.grab_input_using_trait('object_track_set')
            
            print('timestamp = {!r}'.format(timestamp))

            im = get_pil_image(in_img_c.image()).convert('RGB')
            im = np.array(im)[:, :, ::-1].copy()
            
            if not self._ots_input_flag:
                starting_frame_id = 0
            else:
                starting_frame_id = [t.first_frame for t in input_ots.tracks()][-1]
                last_track_id = [t.id for t in input_ots.tracks()][-1]               

            if self._ots_input_flag and not self._init_track_flag and self._step_id == (starting_frame_id):
                starting_frame_id = [t.first_frame for t in input_ots.tracks()][-1]
                last_track_id = [t.id for t in input_ots.tracks()][-1]
                input_track = input_ots.get_track(last_track_id)[starting_frame_id]
                temp_bbox = input_track.detection.bounding_box()
                self._seed_bbox = [temp_bbox.min_x(), temp_bbox.min_y(),
                                   temp_bbox.width(), temp_bbox.height()]
                self._init_track_flag = True
            elif not self._ots_input_flag and not self._init_track_flag and self._step_id == (starting_frame_id):
                self._init_track_flag = True

            test_fcn_begin = timer()
            if self._init_track_flag:
                bbox, self.tracker, score = pysot_step(im, self._step_id, self._seed_bbox,
                                                       self._model_path, self._config_path,
                                                       self.tracker)

                test_fcn_end = timer()
                print('%%%test_fcn feature elapsed time: {}'.format(test_fcn_end - test_fcn_begin))

                fid = timestamp.get_frame()
                ts = timestamp.get_time_usec()

                do_bbox = BoundingBox(bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
                det_obj_set = DetectedObjectSet()
                d_obj = DetectedObject(do_bbox, confidence=score)
                det_obj_set.add(d_obj)

                bbox_center = [bbox[0]+0.5*bbox[2], bbox[1]+0.5*bbox[3]]

                track_state_list = [track_state(frame_id=self._step_id,
                        bbox_center=bbox_center,
                        interaction_feature=None,
                        app_feature=None,
                        bbox=[int(t) for t in bbox],
                        detected_object=d_obj,
                        sys_frame_id=fid, sys_frame_time=ts)]
                next_track_id = int(self._track_set.get_max_track_id()) + 1

                track_idx_list = [track.track_id for track in self._track_set.iter_active()]

                # if there are no tracks, generate new tracks from the track_state_list
                if not self._track_flag:
                    next_track_id = self._track_set.add_new_track_state_list(next_track_id,
                                    track_state_list)
                    self._track_flag = True
                else:
                    self._track_set.update_track(track_idx_list[0], \
                            track_state_list[0])
                    self._track_set.reset_updated_flag()
                
                ot_list = st2ot_list(self._track_set)
                ots = ObjectTrackSet(ot_list)
            
                if score > self._track_th and self._init_track_flag:
                    self.push_to_port_using_trait('object_track_set', ots)
                    
                self.push_to_port_using_trait('detected_object_set', det_obj_set)

            self._step_id += 1

            self._base_step()

        except BaseException as e:
            print( repr( e ) )
            import traceback
            print( traceback.format_exc() )
            #sys.stdout.flush(

# ==================================================================
def __sprokit_register__():
    from sprokit.pipeline import process_factory

    module_name = 'python:kwiver.pytorch.PYSOTTracker'

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process('pysot_tracker',
                                'Siamese tracking using the pysot library',
                                PYSOTTracker)

    process_factory.mark_process_module_as_loaded(module_name)
