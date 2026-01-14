# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
SiamMask visual object tracker implementation.

Uses Siamese networks with mask prediction for robust single-object tracking.
Supports multiple concurrent track instances with detection-based initialization.

This implementation uses the vital track_objects algorithm interface.
"""

import sys
import ast
import logging

import torch
import numpy as np
import scriptconfig as scfg

from timeit import default_timer as timer

from kwiver.vital.algo import TrackObjects
from kwiver.vital.types import Image, ImageContainer
from kwiver.vital.types import BoundingBoxD
from kwiver.vital.types import DetectedObject, DetectedObjectSet
from kwiver.vital.types import ObjectTrackState, Track, ObjectTrackSet

from viame.pytorch.siammask.core.config import cfg
from viame.pytorch.siammask.models.model_builder import ModelBuilder
from viame.pytorch.siammask.tracker.tracker_builder import build_tracker
from viame.pytorch.siammask.utils.bbox import get_axis_aligned_bbox
from viame.pytorch.siammask.utils.model_load import load_pretrain

logger = logging.getLogger(__name__)


def gpu_list_desc(use_for=None):
    """Generate a description for a GPU list config trait."""
    return ('define which GPUs to use{}: "all", "None", or a comma-separated list, e.g. "1,2"'
            .format('' if use_for is None else ' for ' + use_for))


def parse_gpu_list(gpu_list_str):
    """Parse a string representing a list of GPU indices to a list of numeric GPU indices."""
    return ([] if gpu_list_str == 'None' else
            None if gpu_list_str == 'all' else
            list(map(int, gpu_list_str.split(','))))


# =============================================================================
# Configuration
# =============================================================================

class SiamMaskTrackerConfig(scfg.DataConfig):
    """Configuration for SiamMask tracker algorithm."""
    gpu_list = scfg.Value('all', help=gpu_list_desc(use_for='Siamese short-term trackers'))
    config_file = scfg.Value('models/siammask_config.yaml', help='Path to configuration file.')
    model_file = scfg.Value('models/siammask_model.pth', help='Path to trained model file.')
    seed_bbox = scfg.Value('[100, 100, 100, 100]', help='Start bounding box for debug mode only')
    threshold = scfg.Value(0.0, help='Minimum confidence to keep track.')
    init_threshold = scfg.Value(0.10, help='Minimum detection confidence to initialize tracks.')
    init_intersect = scfg.Value(0.10,
                                help='Do not initialize on detections with this percent overlap with existing tracks')
    terminate_after_n = scfg.Value(50, help='Terminate trackers after no hits for N frames')


# =============================================================================
# SiamMask TrackObjects Algorithm
# =============================================================================

class SiamMaskTracker(TrackObjects):
    """
    SiamMask visual object tracker algorithm.

    Uses Siamese networks with mask prediction for robust single-object
    tracking. Supports multiple concurrent track instances with
    detection-based initialization.
    """

    def __init__(self):
        TrackObjects.__init__(self)

        self._config = SiamMaskTrackerConfig()

        # Model
        self._model = None

        # State
        self._trackers = dict()
        self._tracks = dict()
        self._track_init_frames = dict()
        self._track_last_frames = dict()
        self._track_counter = 0
        self._is_first = True
        self._last_ts = None
        self._last_img = None

    def get_configuration(self):
        cfg = super(TrackObjects, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        from viame.pytorch.utilities import vital_config_update
        config = self.get_configuration()
        vital_config_update(config, cfg_in)

        self._config.gpu_list = config.get_value('gpu_list')
        self._config.config_file = config.get_value('config_file')
        self._config.model_file = config.get_value('model_file')
        self._config.seed_bbox = config.get_value('seed_bbox')
        self._config.threshold = float(config.get_value('threshold'))
        self._config.init_threshold = float(config.get_value('init_threshold'))
        self._config.init_intersect = float(config.get_value('init_intersect'))
        self._config.terminate_after_n = int(config.get_value('terminate_after_n'))

        # Parse GPU list
        self._gpu_list = parse_gpu_list(self._config.gpu_list)

        # Load model configuration
        cfg.merge_from_file(self._config.config_file)

        # Initialize model
        self._model = ModelBuilder()
        self._model = load_pretrain(self._model, self._config.model_file).cuda().eval()

        return True

    def check_configuration(self, cfg):
        return True

    def track(self, ts, image, detections):
        """
        Track objects in the current frame.

        Parameters
        ----------
        ts : vital.types.Timestamp
            Timestamp for the frame
        image : vital.types.ImageContainer
            Current frame image
        detections : vital.types.DetectedObjectSet
            Detections in the current frame (used for initialization)

        Returns
        -------
        vital.types.ObjectTrackSet
            Current track set
        """
        if not ts.has_valid_frame():
            raise RuntimeError("Frame timestamps must contain frame IDs")

        logger.debug('SiamMask tracker stepping, timestamp = %r', ts)

        frame_id = ts.get_frame()
        img = image.image().asarray().astype('uint8')

        # Handle image format
        if len(np.shape(img)) > 2 and np.shape(img)[2] == 1:
            img = img[:, :, 0]
        if len(np.shape(img)) == 2:
            img = np.stack((img,) * 3, axis=-1)
        else:
            img = img[:, :, ::-1].copy()  # RGB vs BGR

        frame_boxes = []

        # Track initialization helper
        def initialize_track(tid, cbox, timestamp, image_arr, dot=None):
            bbox = [cbox.min_x(), cbox.min_y(), cbox.width(), cbox.height()]
            cx, cy, w, h = get_axis_aligned_bbox(np.array(bbox))
            start_box = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
            self._trackers[tid] = build_tracker(self._model)
            self._trackers[tid].init(image_arr, start_box)
            if dot is None:
                self._tracks[tid] = [ObjectTrackState(timestamp, cbox, 1.0)]
            else:
                self._tracks[tid] = [ObjectTrackState(timestamp, cbox, 1.0, dot)]
            self._track_init_frames[tid] = timestamp.get_frame()
            self._track_last_frames[tid] = timestamp.get_frame()

        # Update existing tracks
        tids_to_delete = []
        init_track_ids = []

        for tid in self._trackers.keys():
            if tid in init_track_ids:
                continue  # Already processed (initialized) on frame
            tracker_output = self._trackers[tid].track(img)
            bbox = tracker_output['bbox']
            score = tracker_output['best_score']
            if score > self._config.threshold:
                cbox = BoundingBoxD(
                    bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])

                # Create DetectedObject with mask if available
                if 'mask' in tracker_output and tracker_output['mask'] is not None:
                    det = DetectedObject(cbox, score)

                    # Crop mask to bounding box region (mask is relative to bbox)
                    full_mask = tracker_output['mask']
                    x1 = max(0, int(bbox[0]))
                    y1 = max(0, int(bbox[1]))
                    x2 = min(full_mask.shape[1], int(bbox[0] + bbox[2]))
                    y2 = min(full_mask.shape[0], int(bbox[1] + bbox[3]))

                    if x2 > x1 and y2 > y1:
                        # Extract and convert mask to uint8 (threshold at 0.5)
                        relative_mask = (full_mask[y1:y2, x1:x2] > 0.5).astype(np.uint8) * 255
                        det.mask = ImageContainer(Image(relative_mask))

                    new_state = ObjectTrackState(ts, cbox, score, det)
                else:
                    new_state = ObjectTrackState(ts, cbox, score)

                self._tracks[tid].append(new_state)
                self._track_last_frames[tid] = frame_id
                frame_boxes.append(cbox)
            if frame_id > self._track_last_frames[tid] + self._config.terminate_after_n:
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
            intsct_area = max(0, x2 - x1) * max(0, y2 - y1)
            return intsct_area / (max(cbox1.area(), cbox2.area()))

        if detections is not None:
            filtered_dets = detections.select(self._config.init_threshold)
            for det in filtered_dets:
                # Check for overlap
                cbox = det.bounding_box
                overlaps = False
                for obox in frame_boxes:
                    if box_intersect(cbox, obox) > self._config.init_intersect:
                        overlaps = True
                        break
                if overlaps:
                    continue
                # Initialize new track if necessary
                self._track_counter = self._track_counter + 1
                initialize_track(self._track_counter, cbox, ts, img, det.type)
                frame_boxes.append(cbox)

        # Output tracks
        output_tracks = ObjectTrackSet(
            [Track(tid, trk) for tid, trk in self._tracks.items()])

        self._last_ts = ts
        self._last_img = img
        self._is_first = False

        return output_tracks

    def initialize(self, ts, image, seed_detections):
        """Initialize tracking with seed detections."""
        self.reset()

        if seed_detections is not None and len(seed_detections) > 0:
            return self.track(ts, image, seed_detections)
        elif self._config.seed_bbox:
            # Use configured seed bbox for initialization
            seed_bbox = ast.literal_eval(self._config.seed_bbox)
            if not ts.has_valid_frame():
                raise RuntimeError("Frame timestamps must contain frame IDs")

            img = image.image().asarray().astype('uint8')
            if len(np.shape(img)) > 2 and np.shape(img)[2] == 1:
                img = img[:, :, 0]
            if len(np.shape(img)) == 2:
                img = np.stack((img,) * 3, axis=-1)
            else:
                img = img[:, :, ::-1].copy()

            cbox = BoundingBoxD(seed_bbox[0], seed_bbox[1],
                                seed_bbox[0] + seed_bbox[2],
                                seed_bbox[1] + seed_bbox[3])
            cx, cy, w, h = get_axis_aligned_bbox(np.array(seed_bbox))
            start_box = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]

            self._trackers[0] = build_tracker(self._model)
            self._trackers[0].init(img, start_box)
            self._tracks[0] = [ObjectTrackState(ts, cbox, 1.0)]
            self._track_init_frames[0] = ts.get_frame()
            self._track_last_frames[0] = ts.get_frame()

            self._last_ts = ts
            self._last_img = img
            self._is_first = False

            return ObjectTrackSet([Track(0, self._tracks[0])])

        return ObjectTrackSet([])

    def finalize(self):
        """
        Finalize tracking and return all tracks.

        Returns
        -------
        vital.types.ObjectTrackSet
            Final set of all tracks
        """
        return ObjectTrackSet(
            [Track(tid, trk) for tid, trk in self._tracks.items()])

    def reset(self):
        """Reset tracker state for new sequence."""
        self._trackers = dict()
        self._tracks = dict()
        self._track_init_frames = dict()
        self._track_last_frames = dict()
        self._track_counter = 0
        self._is_first = True
        self._last_ts = None
        self._last_img = None


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "siammask"

    if algorithm_factory.has_algorithm_impl_name(
            SiamMaskTracker.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "SiamMask visual object tracker",
        SiamMaskTracker
    )

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
