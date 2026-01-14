# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Structural RNN (SRNN) multi-object tracker implementation.

Uses Siamese networks for appearance features and structural RNNs for
association learning. Supports camera motion compensation via homographies.

This implementation uses the vital track_objects algorithm interface.
"""

import itertools
import sys
import logging

import numpy as np
import scipy as sp
import scipy.optimize
import scriptconfig as scfg

import torch

# Initialize cuDNN early at module import time to prevent
# CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED when running with other CUDA processes
from viame.pytorch.utilities import init_cudnn
init_cudnn()

from timeit import default_timer as timer
from PIL import Image as pilImage

from kwiver.vital.algo import TrackObjects
from kwiver.vital.types import Image
from kwiver.vital.types import DetectedObject, DetectedObjectSet
from kwiver.vital.types import ObjectTrackState, Track, ObjectTrackSet
from kwiver.vital.types import new_descriptor
from kwiver.vital.util.VitalPIL import get_pil_image

from viame.pytorch.utilities import Grid, gpu_list_desc, parse_gpu_list

from viame.pytorch.srnn.track import track_state, track, track_set
from viame.pytorch.srnn.models import Siamese
from viame.pytorch.srnn.srnn_matching import SRNNMatching, RnnType
from viame.pytorch.srnn.siamese_feature_extractor import SiameseFeatureExtractor
from viame.pytorch.srnn.iou_tracker import IOUTracker
from viame.pytorch.srnn.gt_bbox import GTBBox, GTFileType
from viame.pytorch.srnn.models import get_config

logger = logging.getLogger(__name__)

g_config = get_config()


def timing(desc, f):
    """Return f(), printing a message about how long it took"""
    start = timer()
    result = f()
    end = timer()
    print('%%%', desc, ' elapsed time: ', end - start, sep='')
    return result


def groupby(it, key):
    result = {}
    for x in it:
        result.setdefault(key(x), []).append(x)
    return result


def ts2ots(track_set):
    """Convert internal track_set to ObjectTrackSet."""
    ot_list = [Track(id=t.track_id) for t in track_set]

    for idx, t in enumerate(track_set):
        ot = ot_list[idx]
        for ti in t.full_history:
            ot_state = ObjectTrackState(ti.sys_frame_id, ti.sys_frame_time,
                                        ti.detected_object)
            if not ot.append(ot_state):
                logger.warning('Cannot add ObjectTrackState')
    return ObjectTrackSet(ot_list)


def from_homog_f2f(homog_f2f):
    """Take a F2FHomography and return a triple of a 3x3 numpy.ndarray and
    two integers corresponding to the contained homography and the
    from and to IDs, respectively.
    """
    arr = np.array([
        [homog_f2f.get(r, c) for c in range(3)] for r in range(3)
    ])
    return arr, homog_f2f.from_id, homog_f2f.to_id


def transform_homog(homog, point):
    """Transform point (a length-2 array-like) using homog (a 3x3 ndarray)"""
    point = np.asarray(point)
    ones = np.ones(point.shape[:-1] + (1,), dtype=point.dtype)
    point = np.concatenate((point, ones), axis=-1)
    result = np.matmul(homog, point[..., np.newaxis])[..., 0]
    return result[..., :-1] / result[..., -1:]


def transform_homog_bbox(homog, bbox):
    """Given a bbox as [x_min, y_min, width, height], transform it
    according to homog and return the smallest enclosing bbox in the
    same format.
    """
    x_min, y_min, width, height = bbox
    points = [
        [x_min, y_min],
        [x_min, y_min + height],
        [x_min + width, y_min],
        [x_min + width, y_min + height],
    ]
    tpoints = transform_homog(homog, points)
    tx_min, ty_min = tpoints.min(0)
    tx_max, ty_max = tpoints.max(0)
    twidth, theight = tx_max - tx_min, ty_max - ty_min
    return [tx_min, ty_min, twidth, theight]


# =============================================================================
# Configuration
# =============================================================================

class SRNNTrackerConfig(scfg.DataConfig):
    """Configuration for SRNN tracker algorithm."""
    # GPU list
    gpu_list = scfg.Value('all', help=gpu_list_desc(use_for='SRNN tracking'))

    # Siamese model config
    siamese_model_path = scfg.Value('siamese/snapshot_epoch_6.pt', help='Trained PyTorch model.')
    siamese_model_input_size = scfg.Value(224, help='Model input image size')
    siamese_batch_size = scfg.Value(128, help='siamese model processing batch size')

    # Detection thresholds
    detection_select_threshold = scfg.Value(0.0, help='detection select threshold')
    track_initialization_threshold = scfg.Value(0.0, help='track initialization threshold')

    # Target RNN config
    targetRNN_AIM_model_path = scfg.Value('targetRNN_snapshot/App_LSTM_epoch_51.pt',
                                          help='Trained targetRNN PyTorch model.')
    targetRNN_AIM_V_model_path = scfg.Value('targetRNN_AI/App_LSTM_epoch_51.pt',
                                            help='Trained targetRNN AIM with variable input size PyTorch model.')
    targetRNN_batch_size = scfg.Value(256, help='targetRNN model processing batch size')
    targetRNN_normalized_models = scfg.Value(False,
                                             help='If the provided models have a normalization layer')

    # Matching
    similarity_threshold = scfg.Value(0.5, help='similarity threshold.')

    # IOU tracker
    IOU_tracker_flag = scfg.Value(True, help='IOU tracker flag.')
    IOU_accept_threshold = scfg.Value(0.5, help='IOU accept threshold.')
    IOU_reject_threshold = scfg.Value(0.1, help='IOU reject threshold.')

    # Track search
    track_search_threshold = scfg.Value(0.1, help='track search threshold.')

    # Track termination
    terminate_track_threshold = scfg.Value(15,
                                           help='terminate tracking if target lost for this many read-in frames.')
    sys_terminate_track_threshold = scfg.Value(50,
                                               help='terminate tracking if target lost for this many system frames.')

    # GT bbox (for testing)
    MOT_GTbbox_flag = scfg.Value(False, help='MOT GT bbox flag')
    AFRL_GTbbox_flag = scfg.Value(False, help='AFRL GT bbox flag')
    GT_bbox_file_path = scfg.Value('', help='ground truth detection file for testing')

    # Features
    add_features_to_detections = scfg.Value(True,
                                            help='Should we add internally computed features to detections?')

    # Track initialization
    explicit_initialization = scfg.Value(False,
                                         help='If True, only tracks derived from provided track set should be output')
    initialization_overlap_threshold = scfg.Value(0.7,
                                                  help='Max IOU with initializations for additional detections')


# =============================================================================
# SRNN TrackObjects Algorithm
# =============================================================================

class SRNNTracker(TrackObjects):
    """
    Structural RNN multi-object tracker algorithm.

    Uses Siamese networks for appearance features and structural RNNs
    for association learning.
    """

    def __init__(self):
        TrackObjects.__init__(self)

        self._config = SRNNTrackerConfig()

        # State
        self._step_id = 0
        self._track_set = None
        self._app_feature_extractor = None
        self._srnn_matching = None
        self._iou_tracker = None
        self._grid = None
        self._gtbbox_flag = False
        self._m_bbox = None

        # Homography state
        self._homog_ref_to_base = np.identity(3)
        self._homog_ref_id = None
        self._homog_src_to_ref = None

        # Previous frame state
        self._prev_frame = None
        self._prev_inits = {}
        self._prev_fid = None
        self._prev_ts = None
        self._prev_im = None
        self._prev_homog_src_to_base = None
        self._prev_all_dos = None

    def get_configuration(self):
        cfg = super(TrackObjects, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        from viame.pytorch.utilities import vital_config_update
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        self._config.gpu_list = cfg.get_value('gpu_list')
        self._config.siamese_model_path = cfg.get_value('siamese_model_path')
        self._config.siamese_model_input_size = int(cfg.get_value('siamese_model_input_size'))
        self._config.siamese_batch_size = int(cfg.get_value('siamese_batch_size'))
        self._config.detection_select_threshold = float(cfg.get_value('detection_select_threshold'))
        self._config.track_initialization_threshold = float(cfg.get_value('track_initialization_threshold'))
        self._config.targetRNN_AIM_model_path = cfg.get_value('targetRNN_AIM_model_path')
        self._config.targetRNN_AIM_V_model_path = cfg.get_value('targetRNN_AIM_V_model_path')
        self._config.targetRNN_batch_size = int(cfg.get_value('targetRNN_batch_size'))

        targetRNN_normalized = cfg.get_value('targetRNN_normalized_models').lower()
        self._config.targetRNN_normalized_models = targetRNN_normalized in ('true', '1', 'yes')

        self._config.similarity_threshold = float(cfg.get_value('similarity_threshold'))

        iou_flag = cfg.get_value('IOU_tracker_flag').lower()
        self._config.IOU_tracker_flag = iou_flag in ('true', '1', 'yes')
        self._config.IOU_accept_threshold = float(cfg.get_value('IOU_accept_threshold'))
        self._config.IOU_reject_threshold = float(cfg.get_value('IOU_reject_threshold'))

        self._config.track_search_threshold = float(cfg.get_value('track_search_threshold'))
        self._config.terminate_track_threshold = int(cfg.get_value('terminate_track_threshold'))
        self._config.sys_terminate_track_threshold = int(cfg.get_value('sys_terminate_track_threshold'))

        mot_gt_flag = cfg.get_value('MOT_GTbbox_flag').lower()
        self._config.MOT_GTbbox_flag = mot_gt_flag in ('true', '1', 'yes')

        afrl_gt_flag = cfg.get_value('AFRL_GTbbox_flag').lower()
        self._config.AFRL_GTbbox_flag = afrl_gt_flag in ('true', '1', 'yes')

        self._config.GT_bbox_file_path = cfg.get_value('GT_bbox_file_path')

        add_features = cfg.get_value('add_features_to_detections').lower()
        self._config.add_features_to_detections = add_features in ('true', '1', 'yes')

        explicit_init = cfg.get_value('explicit_initialization').lower()
        self._config.explicit_initialization = explicit_init in ('true', '1', 'yes')

        self._config.initialization_overlap_threshold = float(cfg.get_value('initialization_overlap_threshold'))

        # Initialize components
        self._initialize_components()

        return True

    def _initialize_components(self):
        """Initialize tracking components based on configuration."""
        # GPU list
        gpu_list = parse_gpu_list(str(self._config.gpu_list))

        # Siamese feature extractor
        self._app_feature_extractor = SiameseFeatureExtractor(
            self._config.siamese_model_path,
            self._config.siamese_model_input_size,
            self._config.siamese_batch_size,
            gpu_list
        )

        # SRNN matching
        self._srnn_matching = SRNNMatching(
            self._config.targetRNN_AIM_model_path,
            self._config.targetRNN_AIM_V_model_path,
            self._config.targetRNN_normalized_models,
            self._config.targetRNN_batch_size,
            gpu_list,
        )

        # GT bbox handling
        self._gtbbox_flag = self._config.MOT_GTbbox_flag or self._config.AFRL_GTbbox_flag
        if self._gtbbox_flag:
            if self._config.MOT_GTbbox_flag:
                file_format = GTFileType.MOT
            else:
                file_format = GTFileType.AFRL
            self._m_bbox = GTBBox(self._config.GT_bbox_file_path, file_format)

        # IOU tracker
        self._iou_tracker = IOUTracker(
            self._config.IOU_accept_threshold,
            self._config.IOU_reject_threshold
        )

        # Grid feature extractor
        self._grid = Grid()

        # Track set
        self._track_set = track_set()

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
            Detections in the current frame

        Returns
        -------
        vital.types.ObjectTrackSet
            Current track set
        """
        try:
            self._track_step(ts, image, detections)
        except BaseException as e:
            logger.error(repr(e))
            import traceback
            logger.error(traceback.format_exc())
            raise

        # Convert and return track set
        ots = ts2ots(self._track_set)
        self._step_id += 1
        return ots

    def _track_step(self, timestamp, in_img_c, dos_ptr):
        """Perform tracking step."""
        logger.debug('step %d', self._step_id)

        # Get current frame as PIL image
        im = get_pil_image(in_img_c.image()).convert('RGB')

        # Get detection bbox
        if self._gtbbox_flag:
            dos = [DetectedObject(bbox=bbox, confidence=1.)
                   for bbox in self._m_bbox[self._step_id]]
        else:
            dos = dos_ptr.select(self._config.detection_select_threshold)

        # For algorithm interface, we don't have homography input
        homog_src_to_base = np.identity(3)

        # No explicit initializations in algorithm interface
        inits = {}

        all_dos = list(dos)

        if self._gtbbox_flag:
            fid = ts_time = self._step_id
        else:
            fid = timestamp.get_frame()
            ts_time = timestamp.get_time_usec()

        det_obj_set, all_track_state_list = self._convert_detected_objects(
            all_dos, self._step_id, fid, ts_time, im, homog_src_to_base,
        )
        track_state_list = all_track_state_list

        self._step_track_set(fid, track_state_list, [])

        self._prev_inits = inits
        self._prev_frame = timestamp.get_frame()
        self._prev_fid, self._prev_ts = fid, ts_time
        self._prev_im = im
        self._prev_homog_src_to_base = homog_src_to_base
        self._prev_all_dos = all_dos

    def _convert_detected_objects(
            self, dos, frame_id, sys_frame_id, sys_frame_time,
            image, homog_src_to_base, extra_dos=None,
    ):
        """Turn a list of DetectedObjects into a feature-enhanced
        DetectedObjectSet and list of track_states.
        """
        bboxes = [d_obj.bounding_box for d_obj in dos]
        extra_bboxes = None if extra_dos is None else [
            d_obj.bounding_box for d_obj in extra_dos
        ]

        # Interaction features
        grid_feature_list = timing('grid feature', lambda: (
            self._grid(image.size, bboxes, extra_bboxes)))

        # Appearance features (format: pytorch tensor)
        pt_app_features = timing('app feature', lambda: (
            self._app_feature_extractor(image, bboxes)))

        det_obj_set = DetectedObjectSet()
        track_state_list = []

        # Get new track state from new frame and detections
        for bbox, d_obj, grid_feature, app_feature in zip(
                bboxes, dos, grid_feature_list, pt_app_features,
        ):
            if self._config.add_features_to_detections:
                # Store app feature to detected_object
                app_f = new_descriptor(g_config.A_F_num)
                app_f[:] = app_feature.numpy()
                d_obj.set_descriptor(app_f)
            det_obj_set.add(d_obj)

            # Build track state for current bbox for matching
            bbox_as_list = [bbox.min_x(), bbox.min_y(), bbox.width(), bbox.height()]
            cur_ts = track_state(
                frame_id=frame_id,
                bbox_center=bbox.center(),
                ref_point=transform_homog(homog_src_to_base, bbox.center()),
                interaction_feature=grid_feature,
                app_feature=app_feature,
                bbox=[int(x) for x in bbox_as_list],
                ref_bbox=transform_homog_bbox(homog_src_to_base, bbox_as_list),
                detected_object=d_obj,
                sys_frame_id=sys_frame_id,
                sys_frame_time=sys_frame_time,
            )
            track_state_list.append(cur_ts)

        return det_obj_set, track_state_list

    def _step_track_set(self, frame_id, track_state_list, init_track_states):
        """Step self._track_set using the current frame id, the list of track
        states, and an iterable of (track_id, track_state) pairs to
        directly initialize.
        """
        # Check whether we need to terminate a track
        for track in list(self._track_set.iter_active()):
            if (self._step_id - track[-1].frame_id > self._config.terminate_track_threshold
                or frame_id - track[-1].sys_frame_id > self._config.sys_terminate_track_threshold):
                self._track_set.deactivate_track(track)

        # Get a list of the active tracks before directly adding the
        # explicitly initialized ones.
        tracks = list(self._track_set.iter_active())

        # Directly add explicit init tracks
        for tid, ts in init_track_states:
            self._track_set.make_track(tid, on_exist='restart').append(ts)

        next_track_id = int(self._track_set.get_max_track_id()) + 1

        # Call IOU tracker
        if self._config.IOU_tracker_flag:
            tracks, track_state_list = timing('IOU tracking', lambda: (
                self._iou_tracker(tracks, track_state_list)
            ))

        # Estimate similarity matrix
        similarity_mat = timing('SRNN association', lambda: (
            self._srnn_matching(tracks, track_state_list, self._config.track_search_threshold)
        ))

        # Hungarian algorithm
        row_idx_list, col_idx_list = timing('Hungarian algorithm', lambda: (
            sp.optimize.linear_sum_assignment(similarity_mat)
        ))

        # Contains the row associated with each column, or None
        hung_idx_list = [None] * len(track_state_list)
        for r, c in zip(row_idx_list, col_idx_list):
            hung_idx_list[c] = r

        for c, r in enumerate(hung_idx_list):
            if r is None or -similarity_mat[r, c] < self._config.similarity_threshold:
                # Conditionally initialize a new track
                if not self._config.explicit_initialization and (
                        track_state_list[c].detected_object.confidence
                        >= self._config.track_initialization_threshold
                ):
                    track = self._track_set.make_track(next_track_id)
                    track.append(track_state_list[c])
                    next_track_id += 1
            else:
                # Add to existing track
                tracks[r].append(track_state_list[c])

        logger.debug('total tracks %d', len(self._track_set))

    def initialize(self, ts, image, seed_detections):
        """Initialize tracking with optional seed detections."""
        self.reset()

        if seed_detections is not None and len(seed_detections) > 0:
            return self.track(ts, image, seed_detections)

        return ObjectTrackSet([])

    def finalize(self):
        """
        Finalize tracking and return all tracks.

        Returns
        -------
        vital.types.ObjectTrackSet
            Final set of all tracks
        """
        return ts2ots(self._track_set)

    def reset(self):
        """Reset tracker state for new sequence."""
        self._step_id = 0
        self._track_set = track_set()

        # Reset homography state
        self._homog_ref_to_base = np.identity(3)
        self._homog_ref_id = None
        self._homog_src_to_ref = None

        # Reset previous frame state
        self._prev_frame = None
        self._prev_inits = {}
        self._prev_fid = None
        self._prev_ts = None
        self._prev_im = None
        self._prev_homog_src_to_base = None
        self._prev_all_dos = None


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "srnn"

    if algorithm_factory.has_algorithm_impl_name(
            SRNNTracker.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "Structural RNN multi-object tracker",
        SRNNTracker
    )

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
