# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
ByteTrack multi-object tracker implementation.

ByteTrack uses a two-stage association strategy:
1. First match high-confidence detections with tracked objects
2. Then match remaining tracks with low-confidence detections

This approach helps recover objects during occlusion when detection
confidence drops temporarily.

Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
Every Detection Box" (ECCV 2022)
"""

import logging

import numpy as np
import scipy.optimize
import scipy.linalg
import scriptconfig as scfg

from kwiver.vital.algo import TrackObjects
from kwiver.vital.types import ObjectTrackSet, ObjectTrackState, Track

logger = logging.getLogger(__name__)


# =============================================================================
# Kalman Filter for motion prediction
# =============================================================================

class KalmanFilter:
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space:
        x, y, a, h, vx, vy, va, vh

    where (x, y) is the center of the bounding box, a is the aspect ratio,
    h is the height, and v* are the respective velocities.

    Object motion follows a constant velocity model.
    """

    def __init__(self):
        ndim = 4
        dt = 1.0

        # State transition matrix
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Observation matrix
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty weights
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement."""
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
        """Run Kalman filter prediction step."""
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
        """Project state distribution to measurement space."""
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
        """Run Kalman filter correction step."""
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


# =============================================================================
# Track state enumeration
# =============================================================================

class TrackState:
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


# =============================================================================
# Single track representation
# =============================================================================

class STrack:
    """Represents a single tracked object with Kalman filter state."""
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
        self.history = []  # List of (timestamp, DetectedObject)

    @staticmethod
    def next_id():
        STrack._count += 1
        return STrack._count

    @staticmethod
    def reset_id():
        STrack._count = 0

    def activate(self, kalman_filter, frame_id, timestamp):
        """Start a new tracklet."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        if self.detected_object is not None and timestamp is not None:
            self.history.append((timestamp, self.detected_object))

    def re_activate(self, new_track, frame_id, timestamp, new_id=False):
        """Reactivate a previously lost track."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance,
            self.tlwh_to_xyah(new_track.tlwh)
        )
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
        """Update a matched track."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance,
            self.tlwh_to_xyah(new_tlwh)
        )
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.score = new_track.score
        self._tlwh = new_tlwh
        self.detected_object = new_track.detected_object
        if self.detected_object is not None and timestamp is not None:
            self.history.append((timestamp, self.detected_object))

    def predict(self):
        """Propagate the state distribution one step forward."""
        mean_state = self.mean.copy()
        if self.state != TrackState.TRACKED:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        """Predict states for multiple tracks."""
        if len(stracks) == 0:
            return
        for st in stracks:
            st.predict()

    def mark_lost(self):
        """Mark track as lost."""
        self.state = TrackState.LOST

    def mark_removed(self):
        """Mark track as removed."""
        self.state = TrackState.REMOVED

    @property
    def tlwh(self):
        """Get current bounding box in (top-left x, top-left y, w, h) format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Get current bounding box in (x1, y1, x2, y2) format."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to (center x, center y, aspect ratio, height)."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """Convert bounding box from (x1, y1, x2, y2) to (x, y, w, h)."""
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret


# =============================================================================
# IOU computation
# =============================================================================

def iou_batch(bboxes1, bboxes2):
    """Compute IOU between two sets of bounding boxes."""
    bboxes1 = np.atleast_2d(bboxes1)
    bboxes2 = np.atleast_2d(bboxes2)

    if bboxes1.shape[0] == 0 or bboxes2.shape[0] == 0:
        return np.zeros((bboxes1.shape[0], bboxes2.shape[0]))

    # Compute intersections
    xx1 = np.maximum(bboxes1[:, 0:1], bboxes2[:, 0])
    yy1 = np.maximum(bboxes1[:, 1:2], bboxes2[:, 1])
    xx2 = np.minimum(bboxes1[:, 2:3], bboxes2[:, 2])
    yy2 = np.minimum(bboxes1[:, 3:4], bboxes2[:, 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    # Compute areas
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    union = area1[:, np.newaxis] + area2 - inter
    iou = inter / np.maximum(union, 1e-10)

    return iou


def iou_distance(atracks, btracks):
    """Compute cost matrix based on IOU distance."""
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)))

    atlbrs = np.array([track.tlbr for track in atracks])
    btlbrs = np.array([track.tlbr for track in btracks])

    ious = iou_batch(atlbrs, btlbrs)
    cost_matrix = 1 - ious
    return cost_matrix


# =============================================================================
# Linear assignment
# =============================================================================

def linear_assignment(cost_matrix, thresh):
    """Perform linear assignment with threshold."""
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    # Use scipy's linear sum assignment (Hungarian algorithm)
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
# Converters to/from Kwiver types
# =============================================================================

def to_DetectedObject_list(dos):
    """Get a list of the DetectedObjects in a Kwiver DetectedObjectSet."""
    return list(dos)


def get_DetectedObject_bbox_tlwh(do):
    """Get the bounding box of a Kwiver DetectedObject as (x, y, w, h)."""
    bbox = do.bounding_box
    x = bbox.min_x()
    y = bbox.min_y()
    w = bbox.max_x() - bbox.min_x()
    h = bbox.max_y() - bbox.min_y()
    return np.array([x, y, w, h])


def get_DetectedObject_score(do):
    """Get the confidence score of a Kwiver DetectedObject."""
    return do.confidence


def to_ObjectTrackSet(tracks):
    """Create an ObjectTrackSet from a list of STrack objects."""
    result = []
    for track in tracks:
        if not track.history:
            continue
        t = Track(id=track.track_id)
        for ts, do in track.history:
            ots = ObjectTrackState(ts.get_frame(), ts.get_time_usec(), do)
            if not t.append(ots):
                logger.warning("Unsorted input in to_ObjectTrackSet for track %d", track.track_id)
        result.append(t)
    return ObjectTrackSet(result)


# =============================================================================
# ByteTrack Configuration
# =============================================================================

class ByteTrackTrackerConfig(scfg.DataConfig):
    """Configuration for ByteTrack tracker."""
    high_thresh = scfg.Value(0.6, help='Confidence threshold for high-confidence detections (first stage)')
    low_thresh = scfg.Value(0.1, help='Confidence threshold for low-confidence detections (second stage)')
    match_thresh = scfg.Value(0.8, help='IOU threshold for matching (1 - IOU must be below this)')
    track_buffer = scfg.Value(30, help='Number of frames to keep lost tracks before removal')
    new_track_thresh = scfg.Value(0.6, help='Minimum confidence to create new track from unmatched detection')


# =============================================================================
# ByteTrack Algorithm (TrackObjects implementation)
# =============================================================================

class ByteTrackTracker(TrackObjects):
    """
    ByteTrack multi-object tracker.

    Uses two-stage association to robustly track objects through occlusions
    by leveraging both high and low confidence detections.

    Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
    Every Detection Box" (ECCV 2022)
    """

    def __init__(self):
        TrackObjects.__init__(self)
        self._config = ByteTrackTrackerConfig()

        # Internal state
        self._kalman_filter = None
        self._tracked_stracks = []
        self._lost_stracks = []
        self._removed_stracks = []
        self._frame_id = 0

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

        # Convert types
        self._high_thresh = float(self._config.high_thresh)
        self._low_thresh = float(self._config.low_thresh)
        self._match_thresh = float(self._config.match_thresh)
        self._track_buffer = int(self._config.track_buffer)
        self._new_track_thresh = float(self._config.new_track_thresh)

        # Initialize tracker state
        self._kalman_filter = KalmanFilter()
        self._tracked_stracks = []
        self._lost_stracks = []
        self._removed_stracks = []
        self._frame_id = 0
        STrack.reset_id()

        return True

    def check_configuration(self, cfg):
        """Check if the configuration is valid."""
        return True

    def track(self, ts, image, detections):
        """
        Track objects in a new frame.

        Parameters
        ----------
        ts : Timestamp
            Timestamp for the current frame.
        image : ImageContainer
            The input image for the current frame (may be None).
        detections : DetectedObjectSet
            Detected objects from the current frame.

        Returns
        -------
        ObjectTrackSet
            Updated object track set containing all active tracks.
        """
        self._frame_id += 1

        # Convert detections to internal format
        activated_stracks = []
        refind_stracks = []
        det_list = to_DetectedObject_list(detections) if detections else []

        # Create STrack for each detection
        all_detections = []
        for do in det_list:
            tlwh = get_DetectedObject_bbox_tlwh(do)
            score = get_DetectedObject_score(do)
            all_detections.append(STrack(tlwh, score, detected_object=do))

        # Separate detections by confidence
        high_dets = [d for d in all_detections if d.score >= self._high_thresh]
        low_dets = [d for d in all_detections if self._low_thresh <= d.score < self._high_thresh]

        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []
        tracked = []
        for track in self._tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked.append(track)

        # Predict with Kalman filter
        strack_pool = tracked + self._lost_stracks
        STrack.multi_predict(strack_pool)

        # ===== FIRST STAGE: Match high-confidence detections with tracks =====
        dists = iou_distance(strack_pool, high_dets)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self._match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = high_dets[idet]
            if track.state == TrackState.TRACKED:
                track.update(det, self._frame_id, ts)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self._frame_id, ts, new_id=False)
                refind_stracks.append(track)

        # ===== SECOND STAGE: Match remaining tracks with low-confidence dets =====
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.TRACKED]
        dists = iou_distance(r_tracked_stracks, low_dets)
        matches, u_track_second, _ = linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = low_dets[idet]
            track.update(det, self._frame_id, ts)
            activated_stracks.append(track)

        # Mark remaining tracks as lost
        for it in u_track_second:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.LOST:
                track.mark_lost()
                self._lost_stracks.append(track)

        # ===== HANDLE UNCONFIRMED TRACKS =====
        dists = iou_distance(unconfirmed, [high_dets[i] for i in u_detection])
        matches, u_unconfirmed, u_detection_final = linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(high_dets[u_detection[idet]], self._frame_id, ts)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            self._removed_stracks.append(track)

        # ===== CREATE NEW TRACKS from unmatched high-confidence detections =====
        for inew in u_detection_final:
            det = high_dets[u_detection[inew]]
            if det.score >= self._new_track_thresh:
                det.activate(self._kalman_filter, self._frame_id, ts)
                activated_stracks.append(det)

        # ===== UPDATE LOST TRACKS =====
        for track in self._lost_stracks:
            if self._frame_id - track.frame_id > self._track_buffer:
                track.mark_removed()
                self._removed_stracks.append(track)

        # ===== MERGE TRACK LISTS =====
        self._tracked_stracks = [t for t in self._tracked_stracks if t.state == TrackState.TRACKED]
        self._tracked_stracks = list(set(self._tracked_stracks + activated_stracks))
        self._tracked_stracks = list(set(self._tracked_stracks + refind_stracks))

        self._lost_stracks = [t for t in self._lost_stracks if t.state == TrackState.LOST]
        self._lost_stracks = [t for t in self._lost_stracks if t not in self._tracked_stracks]

        # Output all activated tracks with history
        output_tracks = [t for t in self._tracked_stracks if t.is_activated and len(t.history) > 0]
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
        self._kalman_filter = KalmanFilter()
        self._tracked_stracks = []
        self._lost_stracks = []
        self._removed_stracks = []
        self._frame_id = 0
        STrack.reset_id()


# =============================================================================
# Algorithm Registration
# =============================================================================

def __vital_algorithm_register__():
    """Register the ByteTrack algorithm with KWIVER."""
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "bytetrack"

    if algorithm_factory.has_algorithm_impl_name(
            ByteTrackTracker.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "ByteTrack multi-object tracker with two-stage association",
        ByteTrackTracker
    )

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
