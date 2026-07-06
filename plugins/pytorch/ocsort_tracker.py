# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
OC-SORT / Deep OC-SORT multi-object tracker implementation.

OC-SORT (Observation-Centric SORT) improves on SORT/ByteTrack for objects
with nonlinear motion (e.g. fish) by trusting observations over the Kalman
filter's constant-velocity assumption:

1. Observation-Centric Re-update (ORU): when a lost track is re-found,
   virtual observations are interpolated across the occlusion gap and
   replayed through the Kalman filter, removing the error accumulated
   while coasting on the motion model.
2. Observation-Centric Momentum (OCM): a velocity-direction consistency
   term is added to the association cost, computed from past observations
   rather than filter state.
3. Observation-Centric Recovery (OCR): a last-chance association between
   unmatched tracks' last observations and unmatched detections.

Deep OC-SORT additionally fuses appearance (Re-ID) cosine distance into
the association cost and supports camera motion compensation.

References:
  Cao et al., "Observation-Centric SORT: Rethinking SORT for Robust
  Multi-Object Tracking" (CVPR 2023)
  Maggiolino et al., "Deep OC-SORT: Multi-Pedestrian Tracking by
  Adaptive Re-Identification" (ICIP 2023)

This implementation uses the vital track_objects algorithm interface.
"""

import json
import logging
import os

import numpy as np
import scriptconfig as scfg

from kwiver.vital.algo import TrackObjects
from kwiver.vital.types import ObjectTrackSet

from viame.pytorch.botsort_tracker import (
    CameraMotionCompensation, FeatureExtractor, KalmanFilter,
    iou_batch, linear_assignment,
    to_DetectedObject_list, get_DetectedObject_bbox_tlbr,
    get_DetectedObject_score, to_ObjectTrackSet,
)
from viame.pytorch.utilities import report_cuda_errors

logger = logging.getLogger(__name__)


# =============================================================================
# Observation-centric helpers
# =============================================================================

def speed_direction(bbox1, bbox2):
    """Unit direction of center motion from bbox1 to bbox2 (tlbr)."""
    cx1 = (bbox1[0] + bbox1[2]) / 2.0
    cy1 = (bbox1[1] + bbox1[3]) / 2.0
    cx2 = (bbox2[0] + bbox2[2]) / 2.0
    cy2 = (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt(speed[0] ** 2 + speed[1] ** 2) + 1e-6
    return speed / norm


def velocity_direction_cost(tracks, detections):
    """
    Observation-Centric Momentum (OCM) cost term.

    For each track/detection pair, measures the angular difference between
    the track's observation-based velocity direction and the direction
    from the track's previous observation to the detection. Returns a
    matrix in [0, 1] (0 = perfectly consistent direction).
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))

    cost = np.zeros((len(tracks), len(detections)))
    det_boxes = np.array([d.tlbr_obs for d in detections])

    for i, track in enumerate(tracks):
        if track.velocity is None or track.last_observation is None:
            continue
        vy, vx = track.velocity
        prev = track.last_observation
        pcx = (prev[0] + prev[2]) / 2.0
        pcy = (prev[1] + prev[3]) / 2.0

        dcx = (det_boxes[:, 0] + det_boxes[:, 2]) / 2.0
        dcy = (det_boxes[:, 1] + det_boxes[:, 3]) / 2.0
        dy = dcy - pcy
        dx = dcx - pcx
        norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
        dy /= norm
        dx /= norm

        # Angle between track velocity and candidate direction
        cos_sim = np.clip(vy * dy + vx * dx, -1.0, 1.0)
        angle = np.arccos(cos_sim)  # [0, pi]
        cost[i, :] = angle / np.pi

    return cost


def embedding_cost(tracks, features):
    """Cosine distance between track EMA features and detection features."""
    cost = np.ones((len(tracks), len(features)))
    for i, track in enumerate(tracks):
        if track.smooth_feat is None:
            continue
        for j, feat in enumerate(features):
            if feat is None:
                continue
            cost[i, j] = 1 - np.dot(track.smooth_feat, feat)
    return cost


# =============================================================================
# OC-SORT track
# =============================================================================

class OCTrack:
    """
    Single tracked object for OC-SORT.

    Keeps an observation history (for OCM velocity and ORU re-update) in
    addition to the Kalman filter state.
    """

    _count = 0

    def __init__(self, tlbr, score, delta_t=3, feature=None,
                 feat_alpha=0.95, detected_object=None):
        self.kf = None
        self.mean = None
        self.covariance = None

        self.tlbr_obs = np.asarray(tlbr, dtype=np.float64)
        self.score = score
        self.detected_object = detected_object

        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0
        self.time_since_update = 0
        self.hits = 0
        self.is_activated = False
        self.history = []

        # Observation-centric state
        self.delta_t = delta_t
        self.observations = {}       # frame_id -> tlbr
        self.last_observation = None # most recent tlbr observation
        self.velocity = None         # observation-based (vy, vx) unit vector

        # Appearance (Deep OC-SORT)
        self.smooth_feat = None
        self.curr_feat = feature
        self.feat_alpha = feat_alpha

    @staticmethod
    def next_id():
        OCTrack._count += 1
        return OCTrack._count

    @staticmethod
    def reset_id():
        OCTrack._count = 0

    @staticmethod
    def tlbr_to_xyah(tlbr):
        w = tlbr[2] - tlbr[0]
        h = tlbr[3] - tlbr[1]
        return np.array([
            tlbr[0] + w / 2.0, tlbr[1] + h / 2.0, w / max(h, 1e-6), h
        ])

    @staticmethod
    def xyah_to_tlbr(xyah):
        w = xyah[2] * xyah[3]
        h = xyah[3]
        return np.array([
            xyah[0] - w / 2.0, xyah[1] - h / 2.0,
            xyah[0] + w / 2.0, xyah[1] + h / 2.0
        ])

    @property
    def tlbr(self):
        """Current estimate: filter state if available, else observation."""
        if self.mean is None:
            return self.tlbr_obs.copy()
        return self.xyah_to_tlbr(self.mean[:4])

    def activate(self, kf, frame_id, timestamp):
        self.kf = kf
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kf.initiate(
            self.tlbr_to_xyah(self.tlbr_obs))
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.hits = 1
        self.time_since_update = 0
        if frame_id == 1:
            self.is_activated = True

        self._record_observation(frame_id, self.tlbr_obs)

        if self.curr_feat is not None:
            self.smooth_feat = self.curr_feat

        if self.detected_object is not None and timestamp is not None:
            self.history.append((timestamp, self.detected_object))

    def predict(self):
        self.mean, self.covariance = self.kf.predict(
            self.mean, self.covariance)
        self.time_since_update += 1

    def update(self, det, frame_id, timestamp):
        """
        Associate a new observation.

        Applies Observation-Centric Re-update (ORU) when the track was
        coasting: virtual observations are linearly interpolated between
        the last real observation and the current one and replayed
        through the Kalman filter before the real update.
        """
        gap = frame_id - self.frame_id

        if self.last_observation is not None and gap > 1:
            # ORU: replay interpolated virtual observations over the gap
            start_box = self.last_observation
            end_box = det.tlbr_obs
            for k in range(1, gap):
                ratio = k / float(gap)
                virtual = start_box + (end_box - start_box) * ratio
                self.mean, self.covariance = self.kf.update(
                    self.mean, self.covariance,
                    self.tlbr_to_xyah(virtual))
                self.mean, self.covariance = self.kf.predict(
                    self.mean, self.covariance)

        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance,
            self.tlbr_to_xyah(det.tlbr_obs))

        self._record_observation(frame_id, det.tlbr_obs)

        self.frame_id = frame_id
        self.time_since_update = 0
        self.hits += 1
        self.is_activated = True
        self.score = det.score
        self.detected_object = det.detected_object

        self._update_features(det.curr_feat)

        if self.detected_object is not None and timestamp is not None:
            self.history.append((timestamp, self.detected_object))

    def _record_observation(self, frame_id, tlbr):
        """Store observation and refresh observation-based velocity (OCM)."""
        # Velocity from an observation delta_t frames back for stability
        previous = None
        for dt in range(self.delta_t, 0, -1):
            if frame_id - dt in self.observations:
                previous = self.observations[frame_id - dt]
                break
        if previous is None and self.last_observation is not None:
            previous = self.last_observation

        if previous is not None:
            self.velocity = speed_direction(previous, tlbr)

        self.observations[frame_id] = np.asarray(tlbr, dtype=np.float64)
        self.last_observation = np.asarray(tlbr, dtype=np.float64)

    def _update_features(self, feat):
        if feat is None:
            return
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.feat_alpha * self.smooth_feat + \
                (1 - self.feat_alpha) * feat
        self.smooth_feat = self.smooth_feat / \
            (np.linalg.norm(self.smooth_feat) + 1e-12)


# =============================================================================
# Cost computation
# =============================================================================

def iou_cost_tracks(tracks, detections, use_prediction=True):
    """IoU cost between tracks (predicted or last-observed) and detections."""
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))

    if use_prediction:
        tboxes = np.array([t.tlbr for t in tracks])
    else:
        tboxes = np.array([
            t.last_observation if t.last_observation is not None else t.tlbr
            for t in tracks])
    dboxes = np.array([d.tlbr_obs for d in detections])

    return 1 - iou_batch(tboxes, dboxes)


# =============================================================================
# Configuration
# =============================================================================

class OCSORTTrackerConfig(scfg.DataConfig):
    """Configuration for OC-SORT tracker algorithm."""
    high_thresh = scfg.Value(0.6, help='Detection confidence threshold for first-stage matching')
    low_thresh = scfg.Value(0.1, help='Detection confidence threshold for second-stage (BYTE) matching')
    match_thresh = scfg.Value(0.8, help='Association cost threshold for matching')
    ocr_iou_thresh = scfg.Value(0.3, help='Minimum IoU for observation-centric recovery matching')
    track_buffer = scfg.Value(30, help='Number of frames to keep lost tracks')
    new_track_thresh = scfg.Value(0.6, help='Minimum confidence to create new track')
    min_hits = scfg.Value(1, help='Number of associations before a track is output')
    delta_t = scfg.Value(3, help='Frame gap used to compute observation-based velocity (OCM)')
    vdc_weight = scfg.Value(0.2, help='Weight of velocity direction consistency cost (OCM)')
    use_byte = scfg.Value(True, help='Enable BYTE-style second stage on low-score detections')
    use_reid = scfg.Value(False, help='Enable appearance (Deep OC-SORT) cost fusion')
    reid_weight = scfg.Value(0.25, help='Weight of appearance cost when use_reid is enabled')
    feat_ema_alpha = scfg.Value(0.95, help='EMA momentum for appearance feature smoothing')
    use_cmc = scfg.Value(False, help='Enable camera motion compensation')
    model_path = scfg.Value('', help='Path to Re-ID model weights (Deep OC-SORT)')
    params_file = scfg.Value('', help='Optional JSON file of trained parameters overriding the above')


# =============================================================================
# OC-SORT TrackObjects Algorithm
# =============================================================================

class OCSORTTracker(TrackObjects):
    """
    OC-SORT / Deep OC-SORT multi-object tracker algorithm.

    Observation-centric Kalman tracking robust to nonlinear motion, with
    optional appearance fusion and camera motion compensation.
    """

    def __init__(self):
        TrackObjects.__init__(self)

        self._config = OCSORTTrackerConfig()

        self._kalman_filter = None
        self._cmc = None
        self._feature_extractor = None
        self._tracked = []
        self._lost = []
        self._frame_id = 0

    def get_configuration(self):
        cfg = super(TrackObjects, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    @report_cuda_errors("OCSORTTracker initialization")
    def set_configuration(self, cfg_in):
        from viame.pytorch.utilities import vital_config_update, parse_bool
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        c = self._config
        c.high_thresh = float(cfg.get_value('high_thresh'))
        c.low_thresh = float(cfg.get_value('low_thresh'))
        c.match_thresh = float(cfg.get_value('match_thresh'))
        c.ocr_iou_thresh = float(cfg.get_value('ocr_iou_thresh'))
        c.track_buffer = int(cfg.get_value('track_buffer'))
        c.new_track_thresh = float(cfg.get_value('new_track_thresh'))
        c.min_hits = int(cfg.get_value('min_hits'))
        c.delta_t = int(cfg.get_value('delta_t'))
        c.vdc_weight = float(cfg.get_value('vdc_weight'))
        c.use_byte = parse_bool(cfg.get_value('use_byte'))
        c.use_reid = parse_bool(cfg.get_value('use_reid'))
        c.reid_weight = float(cfg.get_value('reid_weight'))
        c.feat_ema_alpha = float(cfg.get_value('feat_ema_alpha'))
        c.use_cmc = parse_bool(cfg.get_value('use_cmc'))
        c.model_path = cfg.get_value('model_path')
        c.params_file = cfg.get_value('params_file')

        # Trained parameter file (produced by the ocsort trainer) overrides
        # any scalar values configured above
        if c.params_file and os.path.exists(c.params_file):
            with open(c.params_file, 'r') as f:
                params = json.load(f)
            for key, value in params.items():
                if hasattr(c, key):
                    setattr(c, key, value)
            print(f"[OCSORT] Loaded trained parameters from {c.params_file}")

        self._kalman_filter = KalmanFilter(
            std_weight_position=float(getattr(c, 'std_weight_position', 1.0 / 20)),
            std_weight_velocity=float(getattr(c, 'std_weight_velocity', 1.0 / 160)),
        )

        self._cmc = CameraMotionCompensation() if c.use_cmc else None

        if c.use_reid:
            self._feature_extractor = FeatureExtractor(model_path=c.model_path)
        else:
            self._feature_extractor = None

        return True

    def check_configuration(self, cfg):
        return True

    def _association_cost(self, tracks, detections):
        """First-stage cost: IoU + OCM velocity consistency (+ appearance)."""
        cost = iou_cost_tracks(tracks, detections)

        if self._config.vdc_weight > 0:
            cost = cost + self._config.vdc_weight * \
                velocity_direction_cost(tracks, detections)

        if self._config.use_reid:
            feats = [d.curr_feat for d in detections]
            cost = cost + self._config.reid_weight * \
                embedding_cost(tracks, feats)

        return cost

    @report_cuda_errors("OCSORTTracker tracking")
    def track(self, ts, image, detections):
        """Track objects in the current frame."""
        self._frame_id += 1
        c = self._config

        np_image = image.asarray() if image is not None else None

        # Camera motion compensation
        if self._cmc is not None and np_image is not None:
            homography = self._cmc.compute_homography(np_image)
            self._apply_cmc(homography)

        # Build detection wrappers
        det_tracks = []
        boxes = []
        for do in to_DetectedObject_list(detections):
            score = get_DetectedObject_score(do)
            if score < c.low_thresh:
                continue
            tlbr = get_DetectedObject_bbox_tlbr(do)
            det_tracks.append(OCTrack(
                tlbr, score, delta_t=c.delta_t,
                feat_alpha=c.feat_ema_alpha, detected_object=do))
            boxes.append(tlbr)

        # Appearance features (Deep OC-SORT)
        if (c.use_reid and self._feature_extractor is not None
                and np_image is not None and len(boxes) > 0):
            features = self._feature_extractor.extract(np_image, boxes)
            for det, feat in zip(det_tracks, features):
                det.curr_feat = feat

        high_dets = [d for d in det_tracks if d.score >= c.high_thresh]
        low_dets = [d for d in det_tracks
                    if c.low_thresh <= d.score < c.high_thresh]

        # Predict all active tracks
        pool = self._tracked + self._lost
        for t in pool:
            t.predict()

        # === FIRST STAGE: high-confidence association (IoU + OCM + ReID) ===
        cost = self._association_cost(pool, high_dets)
        matches, u_track, u_det = linear_assignment(
            cost, thresh=c.match_thresh)

        for it, idet in matches:
            pool[it].update(high_dets[idet], self._frame_id, ts)

        # === SECOND STAGE (BYTE): low-confidence association, IoU only ===
        remaining = [pool[i] for i in u_track]
        if c.use_byte and len(low_dets) > 0:
            still_tracked = [t for t in remaining if t.time_since_update == 1]
            cost = iou_cost_tracks(still_tracked, low_dets)
            matches, u_second, _ = linear_assignment(cost, thresh=0.5)
            for it, idet in matches:
                still_tracked[it].update(low_dets[idet], self._frame_id, ts)
            matched_second = set(
                id(still_tracked[it]) for it, _ in matches)
            remaining = [t for t in remaining
                         if id(t) not in matched_second]

        # === THIRD STAGE (OCR): last observation vs unmatched detections ===
        u_high = [high_dets[i] for i in u_det]
        if len(remaining) > 0 and len(u_high) > 0:
            cost = iou_cost_tracks(remaining, u_high, use_prediction=False)
            matches, u_track_final, u_det_final = linear_assignment(
                cost, thresh=1.0 - c.ocr_iou_thresh)
            for it, idet in matches:
                remaining[it].update(u_high[idet], self._frame_id, ts)
            remaining = [remaining[i] for i in u_track_final]
            u_high = [u_high[i] for i in u_det_final]

        # === New tracks from remaining unmatched high-confidence dets ===
        for det in u_high:
            if det.score >= c.new_track_thresh:
                det.activate(self._kalman_filter, self._frame_id, ts)

        # === Update track lists ===
        all_tracks = pool + [d for d in u_high
                             if d.track_id > 0]
        self._tracked = []
        self._lost = []
        for t in all_tracks:
            if t.track_id == 0:
                continue
            age = self._frame_id - t.frame_id
            if age == 0:
                if t.hits >= c.min_hits:
                    t.is_activated = True
                self._tracked.append(t)
            elif age <= c.track_buffer:
                self._lost.append(t)
            # else: dropped

        output = [t for t in self._tracked
                  if t.is_activated and len(t.history) > 0]
        return to_ObjectTrackSet(output)

    def _apply_cmc(self, homography):
        """Warp track states and observations by the camera homography."""
        if np.allclose(homography, np.eye(3)):
            return

        def warp_point(x, y):
            pt = homography @ np.array([x, y, 1.0])
            return pt[0] / pt[2], pt[1] / pt[2]

        for t in self._tracked + self._lost:
            if t.mean is not None:
                t.mean[0], t.mean[1] = warp_point(t.mean[0], t.mean[1])
            if t.last_observation is not None:
                x1, y1 = warp_point(*t.last_observation[:2])
                x2, y2 = warp_point(*t.last_observation[2:])
                t.last_observation = np.array([x1, y1, x2, y2])

    @report_cuda_errors("OCSORTTracker tracking")
    def initialize(self, ts, image, seed_detections):
        """Initialize tracking with optional seed detections."""
        self.reset()
        if seed_detections is not None and len(seed_detections) > 0:
            return self.track(ts, image, seed_detections)
        return ObjectTrackSet([])

    @report_cuda_errors("OCSORTTracker finalization")
    def finalize(self):
        """Finalize tracking and return all tracks."""
        all_tracks = self._tracked + self._lost
        output = [t for t in all_tracks if len(t.history) > 0]
        return to_ObjectTrackSet(output)

    def reset(self):
        """Reset tracker state for a new sequence."""
        OCTrack.reset_id()
        self._tracked = []
        self._lost = []
        self._frame_id = 0
        if self._cmc is not None:
            self._cmc.reset()


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "ocsort"

    if algorithm_factory.has_algorithm_impl_name(
            OCSORTTracker.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "OC-SORT / Deep OC-SORT observation-centric multi-object tracker",
        OCSORTTracker
    )

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
