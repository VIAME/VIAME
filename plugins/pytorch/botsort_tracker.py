# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
BoT-SORT (Bag of Tricks for SORT) multi-object tracker implementation.

BoT-SORT improves on ByteTrack with several enhancements:
1. Camera Motion Compensation (CMC): Estimates frame-to-frame homography
   to compensate for camera movement before association.
2. IoU-ReID Fusion: Combines IOU distance and appearance (cosine) distance
   for more robust association.
3. Appearance feature EMA update: Smooths appearance features over time.

Particularly useful for underwater footage from moving platforms/ROVs.

Reference: Aharon et al., "BoT-SORT: Robust Associations Multi-Pedestrian
Tracking" (arXiv 2022)
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
# Camera Motion Compensation (CMC)
# =============================================================================

class CameraMotionCompensation:
    """
    Estimates camera motion between frames using sparse optical flow.

    Computes homography transformation to compensate for camera movement.
    """

    def __init__(self, method='sparse_flow'):
        self.method = method
        self.prev_frame = None
        self.prev_keypoints = None

        try:
            import cv2
            self.cv2 = cv2
            self._available = True
        except ImportError:
            self._available = False
            logger.warning("OpenCV not available, CMC disabled")

    def compute_homography(self, frame):
        """
        Compute homography from previous frame to current frame.

        Returns identity if no previous frame or not enough matches.
        """
        if not self._available:
            return np.eye(3)

        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_keypoints = self._detect_keypoints(gray)
            return np.eye(3)

        # Compute optical flow
        if self.prev_keypoints is None or len(self.prev_keypoints) < 10:
            self.prev_frame = gray
            self.prev_keypoints = self._detect_keypoints(gray)
            return np.eye(3)

        try:
            next_pts, status, _ = self.cv2.calcOpticalFlowPyrLK(
                self.prev_frame, gray, self.prev_keypoints, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(self.cv2.TERM_CRITERIA_EPS | self.cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )

            # Filter good matches
            good_prev = self.prev_keypoints[status.flatten() == 1]
            good_next = next_pts[status.flatten() == 1]

            if len(good_prev) < 4:
                self.prev_frame = gray
                self.prev_keypoints = self._detect_keypoints(gray)
                return np.eye(3)

            # Estimate homography with RANSAC
            H, mask = self.cv2.findHomography(good_prev, good_next, self.cv2.RANSAC, 5.0)

            if H is None:
                H = np.eye(3)

        except Exception as e:
            logger.warning(f"CMC error: {e}")
            H = np.eye(3)

        self.prev_frame = gray
        self.prev_keypoints = self._detect_keypoints(gray)

        return H

    def _detect_keypoints(self, gray):
        """Detect keypoints for optical flow tracking."""
        corners = self.cv2.goodFeaturesToTrack(
            gray, maxCorners=1000, qualityLevel=0.01, minDistance=10
        )
        return corners if corners is not None else np.array([])

    def apply_cmc(self, tracks, homography):
        """
        Apply camera motion compensation to track states.

        Transforms predicted track positions using the homography.
        """
        if np.allclose(homography, np.eye(3)):
            return

        for track in tracks:
            if track.mean is None:
                continue

            # Transform center position
            cx, cy = track.mean[0], track.mean[1]
            pt = np.array([cx, cy, 1])
            transformed = homography @ pt
            transformed /= transformed[2]

            track.mean[0] = transformed[0]
            track.mean[1] = transformed[1]


# =============================================================================
# Kalman Filter
# =============================================================================

class KalmanFilter:
    """Standard Kalman filter for bounding box tracking."""

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

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self._project(mean, covariance)

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

    def _project(self, mean, covariance):
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


# =============================================================================
# Re-ID Feature Extractor with EMA
# =============================================================================

class FeatureExtractor:
    """Deep appearance feature extractor for Re-ID."""

    def __init__(self, model_path=None, device='cuda'):
        self.model = None
        self.device = device
        self.model_path = model_path
        self._initialized = False

    def _initialize(self):
        if self._initialized:
            return

        try:
            import torch
            import torchvision.transforms as transforms
            from torchvision.models import resnet18, ResNet18_Weights

            self.torch = torch

            if self.device == 'cuda' and not torch.cuda.is_available():
                self.device = 'cpu'

            if self.model_path and self.model_path.strip():
                self.model = torch.load(self.model_path, map_location=self.device)
            else:
                self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
                self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

            self.model = self.model.to(self.device)
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            self._initialized = True

        except ImportError:
            logger.warning("PyTorch not available, using dummy features")
            self._initialized = True

    def extract(self, image, boxes):
        """Extract appearance features for detections."""
        self._initialize()

        if self.model is None:
            return np.random.randn(len(boxes), 512).astype(np.float32)

        if len(boxes) == 0:
            return np.array([])

        import torch

        crops = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                crops.append(torch.zeros(3, 128, 64))
            else:
                crop = image[y1:y2, x1:x2]
                crops.append(self.transform(crop))

        batch = torch.stack(crops).to(self.device)

        with torch.no_grad():
            features = self.model(batch)
            features = features.view(features.size(0), -1)
            features = torch.nn.functional.normalize(features, dim=1)

        return features.cpu().numpy()


# =============================================================================
# Track state
# =============================================================================

class TrackState:
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


# =============================================================================
# Single track with EMA appearance features
# =============================================================================

class STrack:
    """Single tracked object with EMA appearance feature update."""
    shared_kalman = KalmanFilter()
    _count = 0

    def __init__(self, tlwh, score, feature=None, detected_object=None):
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

        # EMA appearance feature
        self.smooth_feat = None
        self.curr_feat = feature
        self.alpha = 0.9  # EMA momentum

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
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

        # Initialize smooth feature
        if self.curr_feat is not None:
            self.smooth_feat = self.curr_feat

        if self.detected_object is not None and timestamp is not None:
            self.history.append((timestamp, self.detected_object))

    def re_activate(self, new_track, frame_id, timestamp, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance,
            self.tlwh_to_xyah(new_track.tlwh)
        )
        self._update_features(new_track.curr_feat)

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

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance,
            self.tlwh_to_xyah(new_track._tlwh)
        )

        self._update_features(new_track.curr_feat)

        self.state = TrackState.TRACKED
        self.is_activated = True
        self.score = new_track.score
        self._tlwh = new_track._tlwh
        self.detected_object = new_track.detected_object

        if self.detected_object is not None and timestamp is not None:
            self.history.append((timestamp, self.detected_object))

    def _update_features(self, feat):
        """Update smooth feature using EMA."""
        if feat is None:
            return

        self.curr_feat = feat

        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat

        # Normalize
        self.smooth_feat = self.smooth_feat / np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.TRACKED:
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
# Distance metrics with IoU-ReID fusion
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


def embedding_distance(tracks, detections):
    """Compute cosine distance between track and detection features."""
    cost_matrix = np.ones((len(tracks), len(detections)))

    for i, track in enumerate(tracks):
        if track.smooth_feat is None:
            continue

        for j, det in enumerate(detections):
            if det.curr_feat is None:
                continue

            cost_matrix[i, j] = 1 - np.dot(track.smooth_feat, det.curr_feat)

    return cost_matrix


def fuse_iou_reid(iou_cost, reid_cost, iou_weight=0.5):
    """
    Fuse IOU and ReID costs.

    BoT-SORT uses a weighted combination of both metrics.
    """
    return iou_weight * iou_cost + (1 - iou_weight) * reid_cost


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


def get_DetectedObject_bbox_tlbr(do):
    bbox = do.bounding_box
    return np.array([bbox.min_x(), bbox.min_y(), bbox.max_x(), bbox.max_y()])


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
# BoT-SORT core algorithm
# =============================================================================

@Transformer.decorate
def botsort_core(
    high_thresh=0.6,
    low_thresh=0.1,
    match_thresh=0.8,
    track_buffer=30,
    new_track_thresh=0.6,
    use_cmc=True,
    use_reid=True,
    iou_weight=0.5,
    model_path=None,
    feat_ema_alpha=0.9,
):
    """
    BoT-SORT algorithm as a generator-based Transformer.

    Parameters
    ----------
    high_thresh : float
        Confidence threshold for high-confidence detections.
    low_thresh : float
        Confidence threshold for low-confidence detections.
    match_thresh : float
        Distance threshold for matching.
    track_buffer : int
        Number of frames to keep lost tracks.
    new_track_thresh : float
        Minimum confidence to create new track.
    use_cmc : bool
        Enable camera motion compensation.
    use_reid : bool
        Enable Re-ID features for matching.
    iou_weight : float
        Weight for IOU in IoU-ReID fusion (0=only ReID, 1=only IOU).
    model_path : str
        Path to Re-ID model weights.
    feat_ema_alpha : float
        EMA momentum for feature smoothing.
    """
    kalman_filter = KalmanFilter()
    cmc = CameraMotionCompensation() if use_cmc else None
    feature_extractor = FeatureExtractor(model_path=model_path) if use_reid else None

    tracked_stracks = []
    lost_stracks = []
    removed_stracks = []
    frame_id = 0

    output = None
    while True:
        dos, ts, image = yield output
        frame_id += 1

        # Camera motion compensation
        homography = np.eye(3)
        if cmc is not None and image is not None:
            homography = cmc.compute_homography(image)

        det_list = to_DetectedObject_list(dos)

        # Create detections and extract features
        detections = []
        boxes = []

        for do in det_list:
            tlwh = get_DetectedObject_bbox_tlwh(do)
            score = get_DetectedObject_score(do)
            if score < low_thresh:
                continue
            detections.append(STrack(tlwh, score, detected_object=do))
            detections[-1].alpha = feat_ema_alpha
            boxes.append(get_DetectedObject_bbox_tlbr(do))

        # Extract appearance features
        if use_reid and feature_extractor is not None and image is not None and len(boxes) > 0:
            features = feature_extractor.extract(image, boxes)
            for det, feat in zip(detections, features):
                det.curr_feat = feat

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

        # Apply CMC to track predictions
        if cmc is not None:
            cmc.apply_cmc(strack_pool, homography)

        # === FIRST STAGE: High-confidence matching ===
        iou_cost = iou_distance(strack_pool, high_dets)

        if use_reid:
            reid_cost = embedding_distance(strack_pool, high_dets)
            dists = fuse_iou_reid(iou_cost, reid_cost, iou_weight)
        else:
            dists = iou_cost

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

        # === SECOND STAGE: Low-confidence matching (IOU only) ===
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


class botsort_tracker(KwiverProcess):
    """
    BoT-SORT multi-object tracker sprokit process.

    Uses camera motion compensation and IoU-ReID fusion for robust
    tracking from moving platforms.
    """

    def __init__(self, config):
        KwiverProcess.__init__(self, config)

        add_declare_config(self, "high_thresh", "0.6",
            "Detection confidence threshold for first-stage matching")
        add_declare_config(self, "low_thresh", "0.1",
            "Detection confidence threshold for second-stage matching")
        add_declare_config(self, "match_thresh", "0.8",
            "Distance threshold for matching")
        add_declare_config(self, "track_buffer", "30",
            "Number of frames to keep lost tracks")
        add_declare_config(self, "new_track_thresh", "0.6",
            "Minimum confidence to create new track")
        add_declare_config(self, "use_cmc", "true",
            "Enable camera motion compensation")
        add_declare_config(self, "use_reid", "true",
            "Enable Re-ID features for matching")
        add_declare_config(self, "iou_weight", "0.5",
            "Weight for IOU in IoU-ReID fusion (0=only ReID, 1=only IOU)")
        add_declare_config(self, "model_path", "",
            "Path to Re-ID model weights")
        add_declare_config(self, "feat_ema_alpha", "0.9",
            "EMA momentum for feature smoothing")

        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        self.declare_input_port_using_trait('detected_object_set', required)
        self.declare_input_port_using_trait('timestamp', required)
        self.declare_input_port_using_trait('image', optional)

        self.declare_output_port_using_trait('object_track_set', optional)

    def _configure(self):
        self._tracker = botsort_core(
            high_thresh=float(self.config_value('high_thresh')),
            low_thresh=float(self.config_value('low_thresh')),
            match_thresh=float(self.config_value('match_thresh')),
            track_buffer=int(self.config_value('track_buffer')),
            new_track_thresh=float(self.config_value('new_track_thresh')),
            use_cmc=self.config_value('use_cmc').lower() == 'true',
            use_reid=self.config_value('use_reid').lower() == 'true',
            iou_weight=float(self.config_value('iou_weight')),
            model_path=self.config_value('model_path'),
            feat_ema_alpha=float(self.config_value('feat_ema_alpha')),
        )
        self._base_configure()

    def _step(self):
        dos = self.grab_input_using_trait('detected_object_set')
        ts = self.grab_input_using_trait('timestamp')

        try:
            image = self.grab_input_using_trait('image')
            if image is not None:
                image = image.asarray()
        except:
            image = None

        ots = self._tracker.step(dos, ts, image)

        self.push_to_port_using_trait('object_track_set', ots)
        self._base_step()


def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory

    module_name = 'python:viame.processes.pytorch.botsort_tracker'

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process(
        'botsort_tracker',
        'BoT-SORT tracker with CMC and IoU-ReID fusion',
        botsort_tracker,
    )

    process_factory.mark_process_module_as_loaded(module_name)
