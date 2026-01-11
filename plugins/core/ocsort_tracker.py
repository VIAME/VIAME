# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
OC-SORT (Observation-Centric SORT) multi-object tracker implementation.

OC-SORT improves on ByteTrack with three key innovations:
1. Observation-Centric Momentum (OCM): Uses last observed velocity instead
   of predicted velocity during occlusion, improving motion estimation.
2. Observation-Centric Re-Update (ORU): When a lost track is re-found,
   re-updates Kalman filter with virtual trajectory during occlusion.
3. Velocity Direction Consistency: Penalizes associations with inconsistent
   velocity directions for better handling of non-linear motion.

Reference: Cao et al., "Observation-Centric SORT: Rethinking SORT for
Robust Multi-Object Tracking" (CVPR 2023)
"""

from __future__ import division
from __future__ import print_function

import functools
import logging

import numpy as np
import scipy.optimize
import scipy.linalg

from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from kwiver.sprokit.pipeline import process
from kwiver.vital.types import ObjectTrackSet, ObjectTrackState, Track

logger = logging.getLogger(__name__)


# =============================================================================
# Kalman Filter with OC-SORT modifications
# =============================================================================

class KalmanFilter:
    """
    Kalman filter with OC-SORT's Observation-Centric Momentum (OCM).

    Key difference from standard Kalman: stores last observed state
    to use observed velocity during occlusion instead of predicted.
    """

    def __init__(self, std_weight_position=1.0/20, std_weight_velocity=1.0/160):
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
        covariance = np.linalg.multi_dot([
            self._motion_mat, covariance, self._motion_mat.T
        ]) + motion_cov

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
        covariance = np.linalg.multi_dot([
            self._update_mat, covariance, self._update_mat.T
        ])
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot([
            kalman_gain, projected_cov, kalman_gain.T
        ])
        return new_mean, new_covariance

    def multi_predict(self, mean, covariance, delta_t):
        """
        Predict state delta_t frames forward.

        Used for Observation-Centric Re-Update (ORU).
        """
        for _ in range(delta_t):
            mean, covariance = self.predict(mean, covariance)
        return mean, covariance


# =============================================================================
# Track state
# =============================================================================

class TrackState:
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


# =============================================================================
# Single track representation with OCM
# =============================================================================

class STrack:
    """
    Single tracked object with OC-SORT's Observation-Centric Momentum.

    Stores observation history for OCM and ORU.
    """
    shared_kalman = KalmanFilter()
    _count = 0

    def __init__(self, tlwh, score, detected_object=None):
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
        self.observations = {}  # frame_id -> xyah measurement
        self.last_observation = None
        self.velocity = None  # Observed velocity (OCM)

    @staticmethod
    def next_id():
        STrack._count += 1
        return STrack._count

    @staticmethod
    def reset_id():
        STrack._count = 0

    def activate(self, kalman_filter, frame_id, timestamp):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        measurement = self.tlwh_to_xyah(self._tlwh)
        self.mean, self.covariance = self.kalman_filter.initiate(measurement)

        # Store observation
        self.observations[frame_id] = measurement
        self.last_observation = measurement

        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

        if self.detected_object is not None and timestamp is not None:
            self.history.append((timestamp, self.detected_object))

    def re_activate(self, new_track, frame_id, timestamp, new_id=False):
        """
        Reactivate with Observation-Centric Re-Update (ORU).

        When re-found after occlusion, we re-update the Kalman filter
        with a virtual trajectory based on the new observation.
        """
        new_measurement = self.tlwh_to_xyah(new_track.tlwh)

        # ORU: Re-update with virtual trajectory
        if self.last_observation is not None:
            # Interpolate missing observations
            delta_t = frame_id - self.frame_id
            if delta_t > 1:
                # Linear interpolation of position
                for i in range(1, delta_t):
                    alpha = i / delta_t
                    virtual_obs = (1 - alpha) * self.last_observation + alpha * new_measurement
                    self.mean, self.covariance = self.kalman_filter.update(
                        self.mean, self.covariance, virtual_obs
                    )
                    self.mean, self.covariance = self.kalman_filter.predict(
                        self.mean, self.covariance
                    )

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_measurement
        )

        # Update observed velocity
        if self.last_observation is not None:
            delta_t = max(1, frame_id - self.frame_id)
            self.velocity = (new_measurement[:2] - self.last_observation[:2]) / delta_t

        self.observations[frame_id] = new_measurement
        self.last_observation = new_measurement

        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        self._tlwh = new_track._tlwh
        self.detected_object = new_track.detected_object

        if new_id:
            self.track_id = self.next_id()

        if self.detected_object is not None and timestamp is not None:
            self.history.append((timestamp, self.detected_object))

    def update(self, new_track, frame_id, timestamp):
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_measurement = self.tlwh_to_xyah(new_track.tlwh)

        # Update observed velocity (OCM)
        if self.last_observation is not None:
            self.velocity = new_measurement[:2] - self.last_observation[:2]

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_measurement
        )

        self.observations[frame_id] = new_measurement
        self.last_observation = new_measurement

        self.state = TrackState.TRACKED
        self.is_activated = True
        self.score = new_track.score
        self._tlwh = new_track._tlwh
        self.detected_object = new_track.detected_object

        if self.detected_object is not None and timestamp is not None:
            self.history.append((timestamp, self.detected_object))

    def predict(self):
        """
        Predict with Observation-Centric Momentum (OCM).

        Use observed velocity instead of predicted velocity during occlusion.
        """
        mean_state = self.mean.copy()

        # OCM: Use observed velocity if available and track is lost
        if self.state != TrackState.TRACKED and self.velocity is not None:
            mean_state[4:6] = self.velocity[:2]
            mean_state[7] = 0  # No height velocity

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
# Distance metrics with velocity direction consistency
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


def velocity_direction_consistency(tracks, detections, vdc_weight=0.2):
    """
    Compute velocity direction consistency penalty.

    Penalizes associations where the required velocity direction
    is inconsistent with the track's historical velocity direction.
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))

    cost_matrix = np.zeros((len(tracks), len(detections)))

    for i, track in enumerate(tracks):
        if track.velocity is None or np.linalg.norm(track.velocity[:2]) < 1e-5:
            continue

        track_vel = track.velocity[:2]
        track_vel_norm = track_vel / np.linalg.norm(track_vel)

        track_pos = track.mean[:2] if track.mean is not None else track.tlwh_to_xyah(track._tlwh)[:2]

        for j, det in enumerate(detections):
            det_pos = det.tlwh_to_xyah(det._tlwh)[:2]
            required_vel = det_pos - track_pos

            if np.linalg.norm(required_vel) < 1e-5:
                continue

            required_vel_norm = required_vel / np.linalg.norm(required_vel)

            # Cosine similarity (1 = same direction, -1 = opposite)
            cos_sim = np.dot(track_vel_norm, required_vel_norm)

            # Penalize if direction is inconsistent (negative cosine = opposite direction)
            if cos_sim < 0:
                cost_matrix[i, j] = vdc_weight * (1 - cos_sim)

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
# Transformer pattern
# =============================================================================

class Transformer:
    __slots__ = '_gen',

    def __init__(self, gen):
        gen.send(None)
        self._gen = gen

    def step(self, *args):
        return self._gen.send(args)

    @classmethod
    def decorate(cls, f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return cls(f(*args, **kwargs))
        return wrapper


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
# OC-SORT core algorithm
# =============================================================================

@Transformer.decorate
def ocsort_core(
    high_thresh=0.6,
    low_thresh=0.1,
    match_thresh=0.8,
    track_buffer=30,
    new_track_thresh=0.6,
    use_vdc=True,
    vdc_weight=0.2,
    use_oru=True,
):
    """
    OC-SORT algorithm as a generator-based Transformer.

    Parameters
    ----------
    high_thresh : float
        Confidence threshold for high-confidence detections.
    low_thresh : float
        Confidence threshold for low-confidence detections.
    match_thresh : float
        IOU threshold for matching.
    track_buffer : int
        Number of frames to keep lost tracks.
    new_track_thresh : float
        Minimum confidence to create new track.
    use_vdc : bool
        Enable velocity direction consistency.
    vdc_weight : float
        Weight for velocity direction consistency penalty.
    use_oru : bool
        Enable observation-centric re-update.
    """
    kalman_filter = KalmanFilter()
    tracked_stracks = []
    lost_stracks = []
    removed_stracks = []
    frame_id = 0

    output = None
    while True:
        dos, ts = yield output
        frame_id += 1

        det_list = to_DetectedObject_list(dos)

        detections = []
        for do in det_list:
            tlwh = get_DetectedObject_bbox_tlwh(do)
            score = get_DetectedObject_score(do)
            detections.append(STrack(tlwh, score, detected_object=do))

        high_dets = [d for d in detections if d.score >= high_thresh]
        low_dets = [d for d in detections if low_thresh <= d.score < high_thresh]

        activated_stracks = []
        refind_stracks = []

        unconfirmed = []
        tracked = []
        for track in tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked.append(track)

        strack_pool = tracked + lost_stracks
        STrack.multi_predict(strack_pool)

        # === FIRST STAGE: High-confidence matching with VDC ===
        dists = iou_distance(strack_pool, high_dets)

        if use_vdc:
            vdc_cost = velocity_direction_consistency(strack_pool, high_dets, vdc_weight)
            dists = dists + vdc_cost

        matches, u_track, u_detection = linear_assignment(dists, thresh=match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = high_dets[idet]
            if track.state == TrackState.TRACKED:
                track.update(det, frame_id, ts)
                activated_stracks.append(track)
            else:
                # ORU: Re-activate with virtual trajectory update
                if use_oru:
                    track.re_activate(det, frame_id, ts, new_id=False)
                else:
                    track.update(det, frame_id, ts)
                refind_stracks.append(track)

        # === SECOND STAGE: Low-confidence matching ===
        r_tracked_stracks = [strack_pool[i] for i in u_track
                            if strack_pool[i].state == TrackState.TRACKED]
        dists = iou_distance(r_tracked_stracks, low_dets)
        matches, u_track_second, _ = linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = low_dets[idet]
            track.update(det, frame_id, ts)
            activated_stracks.append(track)

        for it in u_track_second:
            track = r_tracked_stracks[it]
            if track.state != TrackState.LOST:
                track.mark_lost()
                lost_stracks.append(track)

        # === Handle unconfirmed tracks ===
        dists = iou_distance(unconfirmed, [high_dets[i] for i in u_detection])
        matches, u_unconfirmed, u_detection_final = linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(high_dets[u_detection[idet]], frame_id, ts)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # === Create new tracks ===
        for inew in u_detection_final:
            det = high_dets[u_detection[inew]]
            if det.score >= new_track_thresh:
                det.activate(kalman_filter, frame_id, ts)
                activated_stracks.append(det)

        # === Update lost tracks ===
        for track in lost_stracks:
            if frame_id - track.frame_id > track_buffer:
                track.mark_removed()
                removed_stracks.append(track)

        # === Merge track lists ===
        tracked_stracks = [t for t in tracked_stracks if t.state == TrackState.TRACKED]
        tracked_stracks = list(set(tracked_stracks + activated_stracks))
        tracked_stracks = list(set(tracked_stracks + refind_stracks))

        lost_stracks = [t for t in lost_stracks if t.state == TrackState.LOST]
        lost_stracks = [t for t in lost_stracks if t not in tracked_stracks]

        output_tracks = [t for t in tracked_stracks if t.is_activated and len(t.history) > 0]
        output = to_ObjectTrackSet(output_tracks)


# =============================================================================
# Sprokit process
# =============================================================================

def add_declare_config(proc, name_key, default, description):
    proc.add_config_trait(name_key, name_key, default, description)
    proc.declare_config_using_trait(name_key)


class ocsort_tracker(KwiverProcess):
    """
    OC-SORT multi-object tracker sprokit process.

    Uses observation-centric momentum and re-update for robust
    tracking through occlusions with improved motion estimation.
    """

    def __init__(self, config):
        KwiverProcess.__init__(self, config)

        add_declare_config(self, "high_thresh", "0.6",
            "Detection confidence threshold for first-stage matching")
        add_declare_config(self, "low_thresh", "0.1",
            "Detection confidence threshold for second-stage matching")
        add_declare_config(self, "match_thresh", "0.8",
            "IOU threshold for matching")
        add_declare_config(self, "track_buffer", "30",
            "Number of frames to keep lost tracks")
        add_declare_config(self, "new_track_thresh", "0.6",
            "Minimum confidence to create new track")
        add_declare_config(self, "use_vdc", "true",
            "Enable velocity direction consistency")
        add_declare_config(self, "vdc_weight", "0.2",
            "Weight for velocity direction consistency penalty")
        add_declare_config(self, "use_oru", "true",
            "Enable observation-centric re-update")

        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        self.declare_input_port_using_trait('detected_object_set', required)
        self.declare_input_port_using_trait('timestamp', required)
        self.declare_output_port_using_trait('object_track_set', optional)

    def _configure(self):
        self._tracker = ocsort_core(
            high_thresh=float(self.config_value('high_thresh')),
            low_thresh=float(self.config_value('low_thresh')),
            match_thresh=float(self.config_value('match_thresh')),
            track_buffer=int(self.config_value('track_buffer')),
            new_track_thresh=float(self.config_value('new_track_thresh')),
            use_vdc=self.config_value('use_vdc').lower() == 'true',
            vdc_weight=float(self.config_value('vdc_weight')),
            use_oru=self.config_value('use_oru').lower() == 'true',
        )
        self._base_configure()

    def _step(self):
        dos = self.grab_input_using_trait('detected_object_set')
        ts = self.grab_input_using_trait('timestamp')

        ots = self._tracker.step(dos, ts)

        self.push_to_port_using_trait('object_track_set', ots)
        self._base_step()


def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory

    module_name = 'python:viame.processes.core.ocsort_tracker'

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process(
        'ocsort_tracker',
        'OC-SORT tracker with observation-centric momentum and re-update',
        ocsort_tracker,
    )

    process_factory.mark_process_module_as_loaded(module_name)
