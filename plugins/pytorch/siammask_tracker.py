# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import ast
import torch

import numpy as np

from timeit import default_timer as timer

from kwiver.sprokit.pipeline import process
from kwiver.sprokit.processes.kwiver_process import KwiverProcess

from kwiver.vital.types import Image
from kwiver.vital.types import BoundingBoxD
from kwiver.vital.types import DetectedObject, DetectedObjectSet
from kwiver.vital.types import ObjectTrackState, Track, ObjectTrackSet

from viame.pytorch.siammask.core.config import cfg
from viame.pytorch.siammask.models.model_builder import ModelBuilder
from viame.pytorch.siammask.tracker.tracker_builder import build_tracker
from viame.pytorch.siammask.utils.bbox import get_axis_aligned_bbox
from viame.pytorch.siammask.utils.model_load import load_pretrain


def gpu_list_desc(use_for=None):
    """Generate a description for a GPU list config trait."""
    return ('define which GPUs to use{}: "all", "None", or a comma-separated list, e.g. "1,2"'
            .format('' if use_for is None else ' for ' + use_for))


def parse_gpu_list(gpu_list_str):
    """Parse a string representing a list of GPU indices to a list of numeric GPU indices."""
    return ([] if gpu_list_str == 'None' else
            None if gpu_list_str == 'all' else
            list(map(int, gpu_list_str.split(','))))


# ------------------------------------------------------------------------------
class SiamMaskTracker(KwiverProcess):
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        # GPU list
        self.add_config_trait("gpu_list", "gpu_list", 'all',
          gpu_list_desc(use_for='Siamese short-term trackers'))
        self.declare_config_using_trait('gpu_list')

        # Config file
        self.add_config_trait("config_file", "config_file",
          'models/siammask_config.yaml', 'Path to configuration file.')
        self.declare_config_using_trait("config_file")

        # Model file
        self.add_config_trait("model_file", "model_file",
          'models/siammask_model.pth', 'Path to trained model file.')
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

        print('SiamMask tracker stepping, timestamp = {!r}'.format(ts))

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
            inits = self.grab_input_using_trait('initializations')
            self._has_init_signals = True
            init_track_pool = [] if inits is None else inits.tracks()
            init_track_ids = []
        elif self.has_input_port_edge_using_trait('detected_object_set'):
            init_track_pool = []
            init_track_ids = []
        elif self._is_first:
            init_track_pool = []
            cbox = BoundingBoxD(self._seed_bbox[0],
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
                initialize_track(trk.id, init_detection.bounding_box,
                  self._last_ts, self._last_img, init_detection.type)
            # This track has an initialization signal for the current frame
            elif trk[trk.last_frame].frame_id == frame_id:
                init_detection = trk[trk.last_frame].detection()
                initialize_track(trk.id, init_detection.bounding_box,
                  ts, img, init_detection.type)
                frame_boxes.append(init_detection.bounding_box)

        # Update existing tracks
        tids_to_delete = []

        for tid in self._trackers.keys():
            if tid in init_track_ids:
                continue # Already processed (initialized) on frame
            tracker_output = self._trackers[tid].track(img)
            bbox = tracker_output['bbox']
            score = tracker_output['best_score']
            if score > self._threshold:
                cbox = BoundingBoxD(
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
                cbox = det.bounding_box
                overlaps = False
                for obox in frame_boxes:
                    if box_intersect(cbox, obox) > self._init_intersect:
                        overlaps = True
                        break
                if overlaps:
                    continue
                # Initialize new track if necessary
                self._track_counter = self._track_counter + 1
                initialize_track(self._track_counter, cbox, ts, img, det.type)

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
    from kwiver.sprokit.pipeline import process_factory

    module_name = 'python:viame.pytorch.SiamMaskTracker'

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process('siammask_tracker',
      'Siamese tracking using the siammask library', SiamMaskTracker)

    process_factory.mark_process_module_as_loaded(module_name)
