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
        """
        Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            The mean vector (8 dimensional) and covariance matrix (8x8) of
            the new track.
        """
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
        """
        Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state.

        Returns
        -------
        (ndarray, ndarray)
            The predicted mean and covariance of the state distribution.
        """
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
        """
        Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            The projected mean and covariance matrix of the given estimate.
        """
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
        """
        Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h).

        Returns
        -------
        (ndarray, ndarray)
            The measurement-corrected state distribution.
        """
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
    """
    Represents a single tracked object with Kalman filter state.
    """
    shared_kalman = KalmanFilter()
    _count = 0

    def __init__(self, tlwh, score, detected_object=None):
        """
        Initialize a track.

        Parameters
        ----------
        tlwh : array-like
            Bounding box in (top-left x, top-left y, width, height) format.
        score : float
            Detection confidence score.
        detected_object : object, optional
            The original Kwiver DetectedObject.
        """
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
    """
    Compute IOU between two sets of bounding boxes.

    Parameters
    ----------
    bboxes1 : ndarray
        N x 4 array of bounding boxes in (x1, y1, x2, y2) format.
    bboxes2 : ndarray
        M x 4 array of bounding boxes in (x1, y1, x2, y2) format.

    Returns
    -------
    ndarray
        N x M IOU matrix.
    """
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
    """
    Compute cost matrix based on IOU distance.

    Parameters
    ----------
    atracks : list
        List of STrack objects.
    btracks : list
        List of STrack objects.

    Returns
    -------
    ndarray
        Cost matrix (1 - IOU).
    """
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
    """
    Perform linear assignment with threshold.

    Parameters
    ----------
    cost_matrix : ndarray
        Cost matrix (N x M).
    thresh : float
        Threshold for valid assignment.

    Returns
    -------
    tuple
        (matches, unmatched_a, unmatched_b)
        matches: list of (a_idx, b_idx) tuples
        unmatched_a: list of unmatched indices from first set
        unmatched_b: list of unmatched indices from second set
    """
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
# Transformer pattern (from simple_homog_tracker.py)
# =============================================================================

class Transformer:
    """A Transformer is a stateful object that receives one tuple of
    values at a time and produces one value at a time in response."""

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
# ByteTrack core algorithm
# =============================================================================

@Transformer.decorate
def bytetrack_core(
    high_thresh=0.6,
    low_thresh=0.1,
    match_thresh=0.8,
    track_buffer=30,
    new_track_thresh=0.6,
):
    """
    ByteTrack algorithm as a generator-based Transformer.

    Parameters
    ----------
    high_thresh : float
        Confidence threshold for high-confidence detections (first stage).
    low_thresh : float
        Confidence threshold for low-confidence detections (second stage).
    match_thresh : float
        IOU threshold for matching (as 1 - IOU cost).
    track_buffer : int
        Number of frames to keep lost tracks before removal.
    new_track_thresh : float
        Minimum confidence to create a new track.

    Yields/receives
    ----------------
    Input: (detected_object_set, timestamp)
    Output: ObjectTrackSet
    """
    kalman_filter = KalmanFilter()
    tracked_stracks = []  # Currently tracked
    lost_stracks = []     # Lost but potentially recoverable
    removed_stracks = []  # Removed (not used, kept for potential future use)
    frame_id = 0

    output = None
    while True:
        dos, ts = yield output
        frame_id += 1

        # Convert detections to internal format
        activated_stracks = []
        refind_stracks = []
        det_list = to_DetectedObject_list(dos)

        # Create STrack for each detection
        detections = []
        for do in det_list:
            tlwh = get_DetectedObject_bbox_tlwh(do)
            score = get_DetectedObject_score(do)
            detections.append(STrack(tlwh, score, detected_object=do))

        # Separate detections by confidence
        high_dets = [d for d in detections if d.score >= high_thresh]
        low_dets = [d for d in detections if low_thresh <= d.score < high_thresh]

        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []
        tracked = []
        for track in tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked.append(track)

        # Predict with Kalman filter
        strack_pool = tracked + lost_stracks
        STrack.multi_predict(strack_pool)

        # ===== FIRST STAGE: Match high-confidence detections with tracks =====
        dists = iou_distance(strack_pool, high_dets)
        matches, u_track, u_detection = linear_assignment(dists, thresh=match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = high_dets[idet]
            if track.state == TrackState.TRACKED:
                track.update(det, frame_id, ts)
                activated_stracks.append(track)
            else:
                track.re_activate(det, frame_id, ts, new_id=False)
                refind_stracks.append(track)

        # ===== SECOND STAGE: Match remaining tracks with low-confidence dets =====
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.TRACKED]
        dists = iou_distance(r_tracked_stracks, low_dets)
        matches, u_track_second, _ = linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = low_dets[idet]
            track.update(det, frame_id, ts)
            activated_stracks.append(track)

        # Mark remaining tracks as lost
        for it in u_track_second:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.LOST:
                track.mark_lost()
                lost_stracks.append(track)

        # ===== HANDLE UNCONFIRMED TRACKS =====
        dists = iou_distance(unconfirmed, [high_dets[i] for i in u_detection])
        matches, u_unconfirmed, u_detection_final = linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(high_dets[u_detection[idet]], frame_id, ts)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # ===== CREATE NEW TRACKS from unmatched high-confidence detections =====
        for inew in u_detection_final:
            det = high_dets[u_detection[inew]]
            if det.score >= new_track_thresh:
                det.activate(kalman_filter, frame_id, ts)
                activated_stracks.append(det)

        # ===== UPDATE LOST TRACKS =====
        for track in lost_stracks:
            if frame_id - track.frame_id > track_buffer:
                track.mark_removed()
                removed_stracks.append(track)

        # ===== MERGE TRACK LISTS =====
        tracked_stracks = [t for t in tracked_stracks if t.state == TrackState.TRACKED]
        tracked_stracks = list(set(tracked_stracks + activated_stracks))
        tracked_stracks = list(set(tracked_stracks + refind_stracks))

        lost_stracks = [t for t in lost_stracks if t.state == TrackState.LOST]
        lost_stracks = [t for t in lost_stracks if t not in tracked_stracks]

        # Output all activated tracks with history
        output_tracks = [t for t in tracked_stracks if t.is_activated and len(t.history) > 0]
        output = to_ObjectTrackSet(output_tracks)


# =============================================================================
# Sprokit process
# =============================================================================

def add_declare_config(proc, name_key, default, description):
    proc.add_config_trait(name_key, name_key, default, description)
    proc.declare_config_using_trait(name_key)


class bytetrack_tracker( KwiverProcess ):
    """
    ByteTrack multi-object tracker sprokit process.

    Uses two-stage association to robustly track objects through occlusions
    by leveraging both high and low confidence detections.
    """

    def __init__(self, config):
        KwiverProcess.__init__(self, config)

        # Declare configuration parameters
        add_declare_config(self, "high_thresh", "0.6",
            "Detection confidence threshold for first-stage matching (high-confidence)")
        add_declare_config(self, "low_thresh", "0.1",
            "Detection confidence threshold for second-stage matching (low-confidence)")
        add_declare_config(self, "match_thresh", "0.8",
            "IOU threshold for matching (1 - IOU must be below this)")
        add_declare_config(self, "track_buffer", "30",
            "Number of frames to keep lost tracks before removal")
        add_declare_config(self, "new_track_thresh", "0.6",
            "Minimum confidence to create new track from unmatched detection")

        # Declare ports
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        # Input ports
        self.declare_input_port_using_trait('detected_object_set', required)
        self.declare_input_port_using_trait('timestamp', required)

        # Output ports
        self.declare_output_port_using_trait('object_track_set', optional)

    def _configure(self):
        self._tracker = bytetrack_core(
            high_thresh=float(self.config_value('high_thresh')),
            low_thresh=float(self.config_value('low_thresh')),
            match_thresh=float(self.config_value('match_thresh')),
            track_buffer=int(self.config_value('track_buffer')),
            new_track_thresh=float(self.config_value('new_track_thresh')),
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

    module_name = 'python:viame.processes.core.bytetrack_tracker'

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process(
        'bytetrack_tracker',
        'ByteTrack multi-object tracker with two-stage association',
        bytetrack_tracker,
    )

    process_factory.mark_process_module_as_loaded(module_name)
