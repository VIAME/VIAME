# ckwg +29
# Copyright 2018-2020 by Kitware, Inc.
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
import ast
import torch

import numpy as np

from timeit import default_timer as timer

from sprokit.pipeline import process
from kwiver.kwiver_process import KwiverProcess

from vital.types import Image
from vital.types import BoundingBox
from vital.types import DetectedObject, DetectedObjectSet
from vital.types import ObjectTrackState, Track, ObjectTrackSet

from kwiver.arrows.pytorch.parse_gpu_list import gpu_list_desc
from kwiver.arrows.pytorch.parse_gpu_list import parse_gpu_list

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain


# ------------------------------------------------------------------------------
class PYSOTTracker(KwiverProcess):
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        # GPU list
        self.add_config_trait("gpu_list", "gpu_list", 'all',
          gpu_list_desc(use_for='Siamese short-term trackers'))
        self.declare_config_using_trait('gpu_list')

        # Config file
        self.add_config_trait("config_file", "config_file",
          'models/pysot_config.yaml', 'Path to configuration file.')
        self.declare_config_using_trait("config_file")

        # Model file
        self.add_config_trait("model_file", "model_file",
          'models/pysot_model.pth', 'Path to trained model file.')
        self.declare_config_using_trait("model_file")

        # General parameters
        self.add_config_trait("seed_bbox", "seed_bbox",
          '[100, 100, 100, 100]', 'Start bounding box for debug mode only')
        self.declare_config_using_trait("seed_bbox")
        
        self.add_config_trait("threshold", "threshold",
          '0.00', 'Minimum confidence to keep track.')
        self.declare_config_using_trait("threshold")

        self.add_config_trait("init_threshold", "init_threshold",
          '0.10', 'Minimum detection confidence to initialize tracks.')
        self.declare_config_using_trait("init_threshold")

        self.add_config_trait("init_intersect", "init_intersect",
          '0.10', 'Do not initialize on any detections with this percent '
          'overlap with any existing other tracks or detection')
        self.declare_config_using_trait("init_intersect")

        self.add_config_trait("terminate_after_n", "terminate_after_n",
          '50', 'Terminate trackers after no hits for N frames')
        self.declare_config_using_trait("terminate_after_n")

        # # PYSOT Configs
        #self.add_config_trait("pysot_model_exemplar_size",
        #  "pysot_model_exemplar_size", '127', 'Model exemplar image size')
        #self.declare_config_using_trait('pysot_model_exemplar_size')
        #
        #self.add_config_trait("pysot_model_instance_size",
        #  "pysot_model_instance_size", '255', 'Model input instance size')
        #self.declare_config_using_trait('pysot_model_instance_size')
        #
        #self.add_config_trait("pysot_batch_size",
        #  "pysot_batch_size", '28', 'pysot model processing batch size')
        # self.declare_config_using_trait('pysot_batch_size')

        # Port Flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        self.add_port_trait("initializations", "object_track_set",
          "Input external object track initializations")

        # Input Ports (Port Name, Flag)
        self.declare_input_port_using_trait('image', required)
        self.declare_input_port_using_trait('timestamp', required)
        self.declare_input_port_using_trait('initializations', optional)
        self.declare_input_port_using_trait('detected_object_set', optional)

        # Output Ports (Port Name, Flag)
        self.declare_output_port_using_trait('timestamp', optional)
        self.declare_output_port_using_trait('object_track_set', optional)

        # Class persistent state variables
        self._trackers = dict()
        self._tracks = dict()
        self._track_init_frames = dict()
        self._track_last_frames = dict()

    # --------------------------------------------------------------------------
    def _configure(self):
        self._gpu_list = parse_gpu_list(self.config_value('gpu_list'))

        self._model_path = self.config_value('model_file')
        self._config_file = self.config_value('config_file')
        self._threshold = float(self.config_value('threshold'))
        self._seed_bbox = ast.literal_eval(self.config_value("seed_bbox"))
        self._init_threshold = float(self.config_value("init_threshold"))
        self._init_intersect = float(self.config_value("init_intersect"))
        self._terminate_after_n = int(self.config_value("terminate_after_n"))

        cfg.merge_from_file(self._config_file)

        self._model = ModelBuilder()
        self._model = load_pretrain(self._model, self._model_path).cuda().eval()
        self._is_first = True
        self._track_counter = 0

        self._base_configure()

    # --------------------------------------------------------------------------
    def _step(self):

        # Retrieval all inputs for this step
        in_img_c = self.grab_input_using_trait('image')
        ts = self.grab_input_using_trait('timestamp')

        if not ts.has_valid_frame():
            raise RuntimeError("Frame timestamps must contain frame IDs")

        print('PYSOT tracker stepping, timestamp = {!r}'.format(ts))

        frame_id = ts.get_frame()
        img = in_img_c.image().asarray().astype('uint8')

        if len(np.shape(img)) > 2 and np.shape(img)[2] == 1:
            img = img[:,:,0]
        if len(np.shape(img)) == 2:
            img = np.stack((img,)*3, axis=-1)
        else:
            img = img[:, :, ::-1].copy() # RGB vs BGR

        # Handle track initialization
        def initialize_track(tid, cbox, ts, image, dot=None):
            bbox = [cbox.min_x(), cbox.min_y(), cbox.width(), cbox.height()]
            cx, cy, w, h = get_axis_aligned_bbox(np.array(bbox))
            start_box = [cx-(w-1)/2, cy-(h-1)/2, w, h]
            self._trackers[tid] = build_tracker(self._model)
            self._trackers[tid].init(image, start_box)
            if dot is None:
                self._tracks[tid] = [ObjectTrackState(ts, cbox, 1.0)]
            else:
                self._tracks[tid] = [ObjectTrackState(ts, cbox, 1.0, dot)]
            self._track_init_frames[tid] = ts.get_frame()
            self._track_last_frames[tid] = ts.get_frame()

        frame_boxes = []

        if self.has_input_port_edge_using_trait('initializations'):
            initializations = self.grab_input_using_trait('initializations')
            self._has_init_signals = True
            init_track_pool = initializations.tracks()
            init_track_ids = []
        elif self.has_input_port_edge_using_trait('detected_object_set'):
            init_track_pool = []
            init_track_ids = []
        elif self._is_first:
            init_track_pool = []
            cbox = BoundingBox(self._seed_bbox[0],
              self._seed_bbox[1], self._seed_bbox[2], self._seed_bbox[3])
            initialize_track(0, cbox, frame_id, img)
            init_track_ids = [0]

        for trk in init_track_pool:
            # Special case, initialize a track on a previous frame
            if not self._is_first and \
              trk[trk.last_frame].frame_id == self._last_ts.get_frame() and \
              ( not trk.id in self._track_init_frames or \
              self._track_init_frames[ trk.id ] < self._last_ts.get_frame() ):
                init_detection = trk[trk.last_frame].detection()
                initialize_track(trk.id, init_detection.bounding_box(),
                  self._last_ts, self._last_img, init_detection.type())
            # This track has an initialization signal for the current frame
            elif trk[trk.last_frame].frame_id == frame_id:
                init_detection = trk[trk.last_frame].detection()
                initialize_track(trk.id, init_detection.bounding_box(),
                  ts, img, init_detection.type())
                frame_boxes.append(bbox)

        # Update existing tracks
        tids_to_delete = []

        for tid in self._trackers.keys():
            if tid in init_track_ids:
                continue # Already processed (initialized) on frame
            tracker_output = self._trackers[tid].track(img)
            bbox = tracker_output['bbox']
            score = tracker_output['best_score']
            if score > self._threshold:
                cbox = BoundingBox(
                  bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
                new_state = ObjectTrackState(ts, cbox, score)
                self._tracks[tid].append(new_state)
                self._track_last_frames[tid] = frame_id
                frame_boxes.append(cbox)
            if frame_id > self._track_last_frames[tid] + self._terminate_after_n:
                tids_to_delete.append(tid)

        for tid in tids_to_delete:
            del self._trackers[tid]
            del self._tracks[tid]
            del self._track_init_frames[tid]
            del self._track_last_frames[tid]

        # Detection-based initialization
        def box_intersect(cbox1, cbox2):
            x1 = max(cbox1.min_x(), cbox2.min_x())
            x2 = min(cbox1.max_x(), cbox2.max_x())
            y1 = max(cbox1.min_y(), cbox2.min_y())
            y2 = min(cbox1.max_y(), cbox2.max_y())
            intsct_area = max(0, x2-x1) * max(0, y2-y1)
            return intsct_area/(max(cbox1.area(), cbox2.area()))

        if self.has_input_port_edge_using_trait('detected_object_set'):
            detections = self.grab_input_using_trait('detected_object_set')
            detections = detections.select(self._init_threshold)
            for det in detections:
                # Check for overlap
                cbox = det.bounding_box()
                overlaps = False
                for obox in frame_boxes:
                    if box_intersect(cbox, obox) > self._init_intersect:
                        overlaps = True
                        break
                if overlaps:
                    continue
                # Initialize new track if necessary
                self._track_counter = self._track_counter + 1
                initialize_track(self._track_counter, cbox, ts, img, det.type())

        # Output tracks
        output_tracks = ObjectTrackSet(
          [Track(tid, trk) for tid, trk in self._tracks.items()])

        self.push_to_port_using_trait('timestamp', ts)
        self.push_to_port_using_trait('object_track_set', output_tracks)

        self._last_ts = ts
        self._last_img = img
        self._is_first = False
        self._base_step()

# ==============================================================================
def __sprokit_register__():
    from sprokit.pipeline import process_factory

    module_name = 'python:kwiver.pytorch.PYSOTTracker'

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process('pysot_tracker',
      'Siamese tracking using the pysot library', PYSOTTracker)

    process_factory.mark_process_module_as_loaded(module_name)
