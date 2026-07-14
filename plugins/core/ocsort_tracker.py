# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
OC-SORT / Deep OC-SORT (Observation-Centric SORT) multi-object tracker.

OC-SORT improves on ByteTrack for objects with nonlinear motion (e.g. fish)
by trusting observations over the constant-velocity Kalman assumption:

1. Observation-Centric Momentum (OCM): velocity is estimated from past
   observations (over a ``delta_t`` window) and used to keep motion
   direction consistent during association.
2. Observation-Centric Re-Update (ORU): when a lost track is re-found,
   virtual observations are interpolated across the occlusion gap and
   replayed through the Kalman filter, removing the drift accumulated
   while coasting on the motion model.
3. Velocity Direction Consistency (VDC): associations that require a
   direction reversal relative to a track's observed velocity are
   penalized.
4. Observation-Centric Recovery (OCR): a last-chance association between
   lost tracks' last observations and otherwise-unmatched detections.

Deep OC-SORT additionally fuses appearance (Re-ID) cosine distance into
the association cost and supports camera motion compensation (CMC). The
appearance and CMC helpers are imported lazily from the pytorch plugin
only when their options are enabled, so a default (motion-only) OC-SORT
run pulls in no torch/pytorch dependency.

References:
  Cao et al., "Observation-Centric SORT: Rethinking SORT for Robust
  Multi-Object Tracking" (CVPR 2023)
  Maggiolino et al., "Deep OC-SORT: Multi-Pedestrian Tracking by
  Adaptive Re-Identification" (ICIP 2023)
"""

import json
import logging
import os

import numpy as np
import scipy.optimize
import scipy.linalg
import scriptconfig as scfg

from kwiver.vital.algo import TrackObjects
from kwiver.vital.types import ObjectTrackSet, ObjectTrackState, Track

logger = logging.getLogger(__name__)


# =============================================================================
# Kalman Filter with OC-SORT modifications
# =============================================================================


class KalmanFilter:
    """Kalman filter with OC-SORT's Observation-Centric Momentum (OCM)."""

    def __init__(self, std_weight_position=1.0 / 20, std_weight_velocity=1.0 / 160):
        ndim = 4
        dt = 1.0

        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = (
            np.linalg.multi_dot([self._motion_mat, covariance, self._motion_mat.T])
            + motion_cov
        )

        return mean, covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            [self._update_mat, covariance, self._update_mat.T]
        )
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            [kalman_gain, projected_cov, kalman_gain.T]
        )
        return new_mean, new_covariance


# =============================================================================
# Track state
# =============================================================================


class TrackState:
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


# =============================================================================
# Single track representation with OCM and optional appearance memory
# =============================================================================


class STrack:
    """Single tracked object with OC-SORT's Observation-Centric Momentum."""

    shared_kalman = KalmanFilter()
    _count = 0

    def __init__(
        self, tlwh, score, detected_object=None, feature=None, feat_alpha=0.9, delta_t=3
    ):
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0
        self.state = TrackState.NEW
        self.frame_id = 0
        self.start_frame = 0
        self.track_id = 0
        self.detected_object = detected_object
        self.history = []

        # OC-SORT: Store observation history for OCM and ORU
        self.observations = {}
        self.last_observation = None
        self.velocity = None
        self.delta_t = delta_t

        # Deep OC-SORT: appearance feature with EMA smoothing
        self.curr_feat = feature
        self.smooth_feat = None
        self.feat_alpha = feat_alpha

    @staticmethod
    def next_id():
        STrack._count += 1
        return STrack._count

    @staticmethod
    def reset_id():
        STrack._count = 0

    def _update_features(self, feat):
        """EMA update of the appearance feature (Deep OC-SORT)."""
        if feat is None:
            return
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = (
                self.feat_alpha * self.smooth_feat + (1 - self.feat_alpha) * feat
            )
        self.smooth_feat = self.smooth_feat / (np.linalg.norm(self.smooth_feat) + 1e-12)

    def _refresh_velocity(self, frame_id, new_measurement):
        """
        Observation-Centric Momentum: estimate velocity from an observation
        ``delta_t`` frames back (falling back to the most recent one) for a
        more stable direction than a single-frame difference.
        """
        previous = None
        used_dt = 1
        for dt in range(self.delta_t, 0, -1):
            if frame_id - dt in self.observations:
                previous = self.observations[frame_id - dt]
                used_dt = dt
                break
        if previous is None:
            previous = self.last_observation
            used_dt = max(1, frame_id - self.frame_id)
        if previous is not None:
            # Per-frame velocity: divide by the actual span to the
            # observation used (not the single-step gap).
            self.velocity = (new_measurement[:2] - previous[:2]) / max(1, used_dt)

    def activate(self, kalman_filter, frame_id, timestamp):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        measurement = self.tlwh_to_xyah(self._tlwh)
        self.mean, self.covariance = self.kalman_filter.initiate(measurement)

        self.observations[frame_id] = measurement
        self.last_observation = measurement

        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

        if self.curr_feat is not None:
            self.smooth_feat = self.curr_feat

        if self.detected_object is not None and timestamp is not None:
            self.history.append((timestamp, self.detected_object))

    def re_activate(self, new_track, frame_id, timestamp, new_id=False):
        """Reactivate with Observation-Centric Re-Update (ORU)."""
        new_measurement = self.tlwh_to_xyah(new_track.tlwh)

        # ORU: Re-update with virtual trajectory across the occlusion gap
        if self.last_observation is not None:
            delta_t = frame_id - self.frame_id
            if delta_t > 1:
                for i in range(1, delta_t):
                    alpha = i / delta_t
                    virtual_obs = (
                        1 - alpha
                    ) * self.last_observation + alpha * new_measurement
                    self.mean, self.covariance = self.kalman_filter.update(
                        self.mean, self.covariance, virtual_obs
                    )
                    self.mean, self.covariance = self.kalman_filter.predict(
                        self.mean, self.covariance
                    )

        self._refresh_velocity(frame_id, new_measurement)

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_measurement
        )

        self.observations[frame_id] = new_measurement
        self.last_observation = new_measurement

        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        self._tlwh = new_track._tlwh
        self.detected_object = new_track.detected_object
        self._update_features(new_track.curr_feat)

        if new_id:
            self.track_id = self.next_id()

        if self.detected_object is not None and timestamp is not None:
            self.history.append((timestamp, self.detected_object))

    def update(self, new_track, frame_id, timestamp):
        self.tracklet_len += 1

        new_measurement = self.tlwh_to_xyah(new_track.tlwh)

        self._refresh_velocity(frame_id, new_measurement)

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_measurement
        )

        self.observations[frame_id] = new_measurement
        self.last_observation = new_measurement
        self.frame_id = frame_id

        self.state = TrackState.TRACKED
        self.is_activated = True
        self.score = new_track.score
        self._tlwh = new_track._tlwh
        self.detected_object = new_track.detected_object
        self._update_features(new_track.curr_feat)

        if self.detected_object is not None and timestamp is not None:
            self.history.append((timestamp, self.detected_object))

    def predict(self):
        """Predict with Observation-Centric Momentum (OCM)."""
        mean_state = self.mean.copy()

        if self.state != TrackState.TRACKED and self.velocity is not None:
            mean_state[4:6] = self.velocity[:2]
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        for st in stracks:
            st.predict()

    def mark_lost(self):
        self.state = TrackState.LOST

    def mark_removed(self):
        self.state = TrackState.REMOVED

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def last_observation_tlbr(self):
        """tlbr of the most recent real observation (xyah), for OCR."""
        if self.last_observation is None:
            return self.tlbr
        cx, cy, a, h = self.last_observation[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret


# =============================================================================
# Distance metrics
# =============================================================================


def iou_batch(bboxes1, bboxes2):
    bboxes1 = np.atleast_2d(bboxes1)
    bboxes2 = np.atleast_2d(bboxes2)

    if bboxes1.shape[0] == 0 or bboxes2.shape[0] == 0:
        return np.zeros((bboxes1.shape[0], bboxes2.shape[0]))

    xx1 = np.maximum(bboxes1[:, 0:1], bboxes2[:, 0])
    yy1 = np.maximum(bboxes1[:, 1:2], bboxes2[:, 1])
    xx2 = np.minimum(bboxes1[:, 2:3], bboxes2[:, 2])
    yy2 = np.minimum(bboxes1[:, 3:4], bboxes2[:, 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    union = area1[:, np.newaxis] + area2 - inter
    return inter / np.maximum(union, 1e-10)


def iou_distance(atracks, btracks):
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)))

    atlbrs = np.array([track.tlbr for track in atracks])
    btlbrs = np.array([track.tlbr for track in btracks])

    return 1 - iou_batch(atlbrs, btlbrs)


def last_observation_iou_distance(tracks, detections):
    """IoU distance using each track's last real observation (for OCR)."""
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))

    atlbrs = np.array([track.last_observation_tlbr for track in tracks])
    btlbrs = np.array([det.tlbr for det in detections])

    return 1 - iou_batch(atlbrs, btlbrs)


def velocity_direction_consistency(tracks, detections, vdc_weight=0.2):
    """Compute velocity direction consistency penalty."""
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))

    cost_matrix = np.zeros((len(tracks), len(detections)))

    for i, track in enumerate(tracks):
        if track.velocity is None or np.linalg.norm(track.velocity[:2]) < 1e-5:
            continue

        track_vel = track.velocity[:2]
        track_vel_norm = track_vel / np.linalg.norm(track_vel)

        track_pos = (
            track.mean[:2]
            if track.mean is not None
            else track.tlwh_to_xyah(track._tlwh)[:2]
        )

        for j, det in enumerate(detections):
            det_pos = det.tlwh_to_xyah(det._tlwh)[:2]
            required_vel = det_pos - track_pos

            if np.linalg.norm(required_vel) < 1e-5:
                continue

            required_vel_norm = required_vel / np.linalg.norm(required_vel)
            cos_sim = np.dot(track_vel_norm, required_vel_norm)

            if cos_sim < 0:
                cost_matrix[i, j] = vdc_weight * (1 - cos_sim)

    return cost_matrix


def embedding_distance(tracks, detections):
    """Cosine appearance distance between track EMA features and detections."""
    cost_matrix = np.ones((len(tracks), len(detections)))

    for i, track in enumerate(tracks):
        if track.smooth_feat is None:
            continue
        for j, det in enumerate(detections):
            if det.curr_feat is None:
                continue
            cost_matrix[i, j] = 1 - np.dot(track.smooth_feat, det.curr_feat)

    return cost_matrix


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_indices, col_indices = scipy.optimize.linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_a = list(range(cost_matrix.shape[0]))
    unmatched_b = list(range(cost_matrix.shape[1]))

    for row, col in zip(row_indices, col_indices):
        if cost_matrix[row, col] <= thresh:
            matches.append((row, col))
            if row in unmatched_a:
                unmatched_a.remove(row)
            if col in unmatched_b:
                unmatched_b.remove(col)

    return matches, unmatched_a, unmatched_b


# =============================================================================
# Converters
# =============================================================================


def to_DetectedObject_list(dos):
    return list(dos)


def get_DetectedObject_bbox_tlwh(do):
    bbox = do.bounding_box
    x = bbox.min_x()
    y = bbox.min_y()
    w = bbox.max_x() - bbox.min_x()
    h = bbox.max_y() - bbox.min_y()
    return np.array([x, y, w, h])


def get_DetectedObject_score(do):
    return do.confidence


def to_ObjectTrackSet(tracks):
    result = []
    for track in tracks:
        if not track.history:
            continue
        t = Track(id=track.track_id)
        for ts, do in track.history:
            ots = ObjectTrackState(ts.get_frame(), ts.get_time_usec(), do)
            if not t.append(ots):
                logger.warning("Unsorted input for track %d", track.track_id)
        result.append(t)
    return ObjectTrackSet(result)


# =============================================================================
# OC-SORT Configuration
# =============================================================================


class OCSORTTrackerConfig(scfg.DataConfig):
    """Configuration for OC-SORT / Deep OC-SORT tracker."""

    high_thresh = scfg.Value(
        0.6, help="Confidence threshold for high-confidence detections"
    )
    low_thresh = scfg.Value(
        0.1, help="Confidence threshold for low-confidence detections"
    )
    match_thresh = scfg.Value(
        0.8, help="Association cost threshold for first-stage matching"
    )
    track_buffer = scfg.Value(30, help="Number of frames to keep lost tracks")
    new_track_thresh = scfg.Value(0.6, help="Minimum confidence to create new track")
    min_hits = scfg.Value(1, help="Number of associations before a track is output")
    delta_t = scfg.Value(
        3, help="Frame gap used to compute observation-based velocity (OCM)"
    )
    use_vdc = scfg.Value(True, help="Enable velocity direction consistency (OCM)")
    vdc_weight = scfg.Value(
        0.2, help="Weight for velocity direction consistency penalty"
    )
    use_oru = scfg.Value(True, help="Enable observation-centric re-update")
    use_byte = scfg.Value(
        True, help="Enable low-confidence (BYTE) second-stage matching"
    )
    use_ocr = scfg.Value(
        True, help="Enable observation-centric recovery of lost tracks"
    )
    ocr_iou_thresh = scfg.Value(
        0.3, help="Minimum IoU for observation-centric recovery matching"
    )
    use_reid = scfg.Value(
        False,
        help="Enable appearance (Deep OC-SORT) cost fusion; imports torch only when enabled",
    )
    reid_weight = scfg.Value(
        0.25, help="Weight of appearance cost when use_reid is enabled"
    )
    feat_ema_alpha = scfg.Value(
        0.9, help="EMA momentum for appearance feature smoothing"
    )
    model_path = scfg.Value("", help="Path to Re-ID model weights (Deep OC-SORT)")
    use_cmc = scfg.Value(False, help="Enable camera motion compensation")
    params_file = scfg.Value(
        "", help="Optional JSON file of trained parameters overriding the above"
    )


# =============================================================================
# OC-SORT Algorithm (TrackObjects implementation)
# =============================================================================


class OCSORTTracker(TrackObjects):
    """
    OC-SORT / Deep OC-SORT multi-object tracker.

    Observation-centric Kalman tracking robust to nonlinear motion, with
    optional appearance fusion (Deep OC-SORT) and camera motion
    compensation. Appearance/CMC helpers are imported lazily so a
    motion-only run needs no torch/pytorch dependency.

    Reference: Cao et al., "Observation-Centric SORT: Rethinking SORT for
    Robust Multi-Object Tracking" (CVPR 2023)
    """

    def __init__(self):
        TrackObjects.__init__(self)
        self._config = OCSORTTrackerConfig()

        # Internal state
        self._kalman_filter = None
        self._cmc = None
        self._feature_extractor = None
        self._tracked_stracks = []
        self._lost_stracks = []
        self._removed_stracks = []
        self._frame_id = 0

        # Kalman noise weights (overridable via params_file)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def get_configuration(self):
        """Get the algorithm configuration."""
        cfg = super(TrackObjects, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        """Set the algorithm configuration."""
        from viame.pytorch.utilities import vital_config_update

        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._config.keys():
            self._config[key] = str(cfg.get_value(key))

        def as_bool(v):
            return str(v).lower() in ("true", "1", "yes")

        # Convert types
        self._high_thresh = float(self._config.high_thresh)
        self._low_thresh = float(self._config.low_thresh)
        self._match_thresh = float(self._config.match_thresh)
        self._track_buffer = int(self._config.track_buffer)
        self._new_track_thresh = float(self._config.new_track_thresh)
        self._min_hits = int(self._config.min_hits)
        self._delta_t = int(self._config.delta_t)
        self._use_vdc = as_bool(self._config.use_vdc)
        self._vdc_weight = float(self._config.vdc_weight)
        self._use_oru = as_bool(self._config.use_oru)
        self._use_byte = as_bool(self._config.use_byte)
        self._use_ocr = as_bool(self._config.use_ocr)
        self._ocr_iou_thresh = float(self._config.ocr_iou_thresh)
        self._use_reid = as_bool(self._config.use_reid)
        self._reid_weight = float(self._config.reid_weight)
        self._feat_ema_alpha = float(self._config.feat_ema_alpha)
        self._model_path = str(self._config.model_path)
        self._use_cmc = as_bool(self._config.use_cmc)

        # Trained parameter file (produced by the ocsort trainer) overrides
        # any scalar values configured above.
        params_file = str(self._config.params_file)
        if params_file and os.path.exists(params_file):
            with open(params_file, "r") as f:
                params = json.load(f)
            self._apply_trained_params(params)
            print(f"[OCSORT] Loaded trained parameters from {params_file}")

        # Initialize tracker state
        self._kalman_filter = KalmanFilter(
            std_weight_position=self._std_weight_position,
            std_weight_velocity=self._std_weight_velocity,
        )
        self._tracked_stracks = []
        self._lost_stracks = []
        self._removed_stracks = []
        self._frame_id = 0
        STrack.reset_id()

        # Optional camera motion compensation (numpy/opencv only, no torch)
        if self._use_cmc:
            from viame.pytorch.botsort_tracker import CameraMotionCompensation

            self._cmc = CameraMotionCompensation()
        else:
            self._cmc = None

        # Optional Deep OC-SORT appearance model. FeatureExtractor imports
        # torch lazily (only on first extract), so torch is never pulled in
        # unless use_reid is enabled AND tracking actually runs.
        if self._use_reid:
            from viame.pytorch.botsort_tracker import FeatureExtractor

            self._feature_extractor = FeatureExtractor(model_path=self._model_path)
        else:
            self._feature_extractor = None

        return True

    def _apply_trained_params(self, params):
        """Override scalar parameters from a trained params JSON dict."""
        mapping = {
            "high_thresh": "_high_thresh",
            "low_thresh": "_low_thresh",
            "match_thresh": "_match_thresh",
            "track_buffer": "_track_buffer",
            "new_track_thresh": "_new_track_thresh",
            "min_hits": "_min_hits",
            "delta_t": "_delta_t",
            "vdc_weight": "_vdc_weight",
            "ocr_iou_thresh": "_ocr_iou_thresh",
            "reid_weight": "_reid_weight",
            "feat_ema_alpha": "_feat_ema_alpha",
            "std_weight_position": "_std_weight_position",
            "std_weight_velocity": "_std_weight_velocity",
        }
        for key, attr in mapping.items():
            if key in params:
                setattr(self, attr, type(getattr(self, attr))(params[key]))

        bool_mapping = {
            "use_vdc": "_use_vdc",
            "use_oru": "_use_oru",
            "use_byte": "_use_byte",
            "use_ocr": "_use_ocr",
        }
        for key, attr in bool_mapping.items():
            if key in params:
                setattr(self, attr, bool(params[key]))

    def check_configuration(self, cfg):
        """Check if the configuration is valid."""
        return True

    def _extract_features(self, np_image, detections):
        """Attach appearance features to detection STracks (Deep OC-SORT)."""
        if self._feature_extractor is None or np_image is None or len(detections) == 0:
            return
        boxes = [d.tlbr for d in detections]
        features = self._feature_extractor.extract(np_image, boxes)
        for det, feat in zip(detections, features):
            det.curr_feat = feat

    def _first_stage_cost(self, strack_pool, high_dets):
        """IoU + optional VDC + optional appearance cost for stage 1."""
        dists = iou_distance(strack_pool, high_dets)
        if self._use_vdc:
            dists = dists + velocity_direction_consistency(
                strack_pool, high_dets, self._vdc_weight
            )
        if self._use_reid:
            dists = dists + self._reid_weight * embedding_distance(
                strack_pool, high_dets
            )
        return dists

    def track(self, ts, image, detections):
        """Track objects in a new frame."""
        self._frame_id += 1

        np_image = image.asarray() if image is not None else None

        det_list = to_DetectedObject_list(detections) if detections else []

        all_detections = []
        for do in det_list:
            tlwh = get_DetectedObject_bbox_tlwh(do)
            score = get_DetectedObject_score(do)
            all_detections.append(
                STrack(
                    tlwh,
                    score,
                    detected_object=do,
                    feat_alpha=self._feat_ema_alpha,
                    delta_t=self._delta_t,
                )
            )

        # Deep OC-SORT: appearance features for all detections
        if self._use_reid:
            self._extract_features(np_image, all_detections)

        high_dets = [d for d in all_detections if d.score >= self._high_thresh]
        low_dets = [
            d for d in all_detections if self._low_thresh <= d.score < self._high_thresh
        ]

        activated_stracks = []
        refind_stracks = []

        unconfirmed = []
        tracked = []
        for track in self._tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked.append(track)

        strack_pool = tracked + self._lost_stracks
        STrack.multi_predict(strack_pool)

        # Camera motion compensation: warp predicted track states
        if self._cmc is not None and np_image is not None:
            homography = self._cmc.compute_homography(np_image)
            self._cmc.apply_cmc(strack_pool, homography)

        # === FIRST STAGE: High-confidence matching (IoU + VDC + ReID) ===
        dists = self._first_stage_cost(strack_pool, high_dets)
        matches, u_track, u_detection = linear_assignment(
            dists, thresh=self._match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = high_dets[idet]
            if track.state == TrackState.TRACKED:
                track.update(det, self._frame_id, ts)
                activated_stracks.append(track)
            else:
                if self._use_oru:
                    track.re_activate(det, self._frame_id, ts, new_id=False)
                else:
                    track.update(det, self._frame_id, ts)
                refind_stracks.append(track)

        # === SECOND STAGE: Low-confidence (BYTE) matching, IoU only ===
        if self._use_byte:
            r_tracked_stracks = [
                strack_pool[i]
                for i in u_track
                if strack_pool[i].state == TrackState.TRACKED
            ]
            dists = iou_distance(r_tracked_stracks, low_dets)
            matches, u_track_second, _ = linear_assignment(dists, thresh=0.5)

            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = low_dets[idet]
                track.update(det, self._frame_id, ts)
                activated_stracks.append(track)

            for it in u_track_second:
                track = r_tracked_stracks[it]
                if track.state != TrackState.LOST:
                    track.mark_lost()
                    self._lost_stracks.append(track)
        else:
            for i in u_track:
                track = strack_pool[i]
                if track.state == TrackState.TRACKED:
                    track.mark_lost()
                    self._lost_stracks.append(track)

        # === THIRD STAGE (OCR): recover lost tracks via last observation ===
        if self._use_ocr:
            lost_pool = [
                strack_pool[i]
                for i in u_track
                if strack_pool[i].state == TrackState.LOST
            ]
            remaining_high = [high_dets[i] for i in u_detection]
            if len(lost_pool) > 0 and len(remaining_high) > 0:
                ocr_dists = last_observation_iou_distance(lost_pool, remaining_high)
                ocr_matches, _, _ = linear_assignment(
                    ocr_dists, thresh=1.0 - self._ocr_iou_thresh
                )
                matched_rem = set()
                for il, ir in ocr_matches:
                    track = lost_pool[il]
                    det = remaining_high[ir]
                    if self._use_oru:
                        track.re_activate(det, self._frame_id, ts, new_id=False)
                    else:
                        track.update(det, self._frame_id, ts)
                    refind_stracks.append(track)
                    matched_rem.add(ir)
                # Keep only detections OCR did not consume
                u_detection = [
                    u_detection[ir]
                    for ir in range(len(remaining_high))
                    if ir not in matched_rem
                ]

        # === Handle unconfirmed tracks ===
        remaining_dets = [high_dets[i] for i in u_detection]
        dists = iou_distance(unconfirmed, remaining_dets)
        matches, u_unconfirmed, u_detection_final = linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(remaining_dets[idet], self._frame_id, ts)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            self._removed_stracks.append(track)

        # === Create new tracks ===
        for inew in u_detection_final:
            det = remaining_dets[inew]
            if det.score >= self._new_track_thresh:
                det.activate(self._kalman_filter, self._frame_id, ts)
                activated_stracks.append(det)

        # === Update lost tracks ===
        for track in self._lost_stracks:
            if self._frame_id - track.frame_id > self._track_buffer:
                track.mark_removed()
                self._removed_stracks.append(track)

        # === Merge track lists ===
        self._tracked_stracks = [
            t for t in self._tracked_stracks if t.state == TrackState.TRACKED
        ]
        self._tracked_stracks = list(set(self._tracked_stracks + activated_stracks))
        self._tracked_stracks = list(set(self._tracked_stracks + refind_stracks))

        self._lost_stracks = [
            t for t in self._lost_stracks if t.state == TrackState.LOST
        ]
        self._lost_stracks = [
            t for t in self._lost_stracks if t not in self._tracked_stracks
        ]

        output_tracks = [
            t
            for t in self._tracked_stracks
            if t.is_activated
            and len(t.history) > 0
            and t.tracklet_len + 1 >= self._min_hits
        ]
        return to_ObjectTrackSet(output_tracks)

    def initialize(self, ts, image, seed_detections):
        """Initialize the tracker for a new sequence."""
        self.reset()
        if seed_detections is not None and len(seed_detections) > 0:
            return self.track(ts, image, seed_detections)
        return ObjectTrackSet([])

    def finalize(self):
        """Finalize tracking and return all tracks."""
        output_tracks = [t for t in self._tracked_stracks if len(t.history) > 0]
        output_tracks += [t for t in self._lost_stracks if len(t.history) > 0]
        return to_ObjectTrackSet(output_tracks)

    def reset(self):
        """Reset the tracker state."""
        self._kalman_filter = KalmanFilter(
            std_weight_position=self._std_weight_position,
            std_weight_velocity=self._std_weight_velocity,
        )
        self._tracked_stracks = []
        self._lost_stracks = []
        self._removed_stracks = []
        self._frame_id = 0
        STrack.reset_id()
        if self._cmc is not None:
            self._cmc.reset()


# =============================================================================
# Algorithm Registration
# =============================================================================


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        OCSORTTracker,
        "ocsort",
        "OC-SORT / Deep OC-SORT tracker with observation-centric momentum, "
        "re-update, recovery, and optional appearance fusion",
    )
