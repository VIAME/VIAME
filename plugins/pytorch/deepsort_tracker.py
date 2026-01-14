# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
DeepSORT multi-object tracker implementation.

DeepSORT extends SORT with deep appearance features for re-identification:
1. Kalman filter for motion prediction
2. Deep CNN for appearance feature extraction
3. Cascade matching: appearance similarity first, then IOU
4. Hungarian algorithm for association

This helps recover object identity after occlusions by matching
appearance features in addition to motion/position.

Reference: Wojke et al., "Simple Online and Realtime Tracking with
a Deep Association Metric" (ICIP 2017)
"""

import logging

import numpy as np
import scipy.optimize
import scipy.linalg
import scipy.stats
import scriptconfig as scfg

from kwiver.vital.algo import TrackObjects
from kwiver.vital.types import ObjectTrackSet, ObjectTrackState, Track

logger = logging.getLogger(__name__)


# =============================================================================
# Kalman Filter for motion prediction
# =============================================================================

class KalmanFilter:
    """Kalman filter for tracking bounding boxes in image space."""

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

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """Compute gating distance (Mahalanobis) between state and measurements."""
        projected_mean, projected_cov = self.project(mean, covariance)

        if only_position:
            projected_mean = projected_mean[:2]
            projected_cov = projected_cov[:2, :2]
            measurements = measurements[:, :2]

        chol_factor = np.linalg.cholesky(projected_cov)
        d = measurements - projected_mean
        z = scipy.linalg.solve_triangular(
            chol_factor, d.T, lower=True, check_finite=False
        )
        return np.sum(z * z, axis=0)


# =============================================================================
# Track state enumeration
# =============================================================================

class TrackState:
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3


# =============================================================================
# Re-ID Feature Extractor
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
# Single track representation
# =============================================================================

class STrack:
    """Represents a single tracked object with Kalman filter state and appearance."""
    _count = 0

    def __init__(self, tlwh, score, feature=None, detected_object=None):
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.score = score
        self.state = TrackState.TENTATIVE
        self.frame_id = 0
        self.start_frame = 0
        self.track_id = 0
        self.detected_object = detected_object
        self.history = []
        self.hits = 1
        self.time_since_update = 0

        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = 3
        self._max_age = 30

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
        self.state = TrackState.TENTATIVE
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.time_since_update = 0
        if self.detected_object is not None and timestamp is not None:
            self.history.append((timestamp, self.detected_object))

    def predict(self):
        self.mean, self.covariance = self.kalman_filter.predict(
            self.mean, self.covariance
        )
        self.time_since_update += 1

    def update(self, detection, frame_id, timestamp):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance,
            self.tlwh_to_xyah(detection._tlwh)
        )
        self._tlwh = detection._tlwh
        self.score = detection.score
        self.detected_object = detection.detected_object

        if detection.features:
            self.features.append(detection.features[0])
            if len(self.features) > 100:
                self.features = self.features[-100:]

        self.hits += 1
        self.time_since_update = 0
        self.frame_id = frame_id

        if self.state == TrackState.TENTATIVE and self.hits >= self._n_init:
            self.state = TrackState.CONFIRMED

        if self.detected_object is not None and timestamp is not None:
            self.history.append((timestamp, self.detected_object))

    def mark_missed(self):
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        elif self.time_since_update > self._max_age:
            self.state = TrackState.DELETED

    def is_tentative(self):
        return self.state == TrackState.TENTATIVE

    def is_confirmed(self):
        return self.state == TrackState.CONFIRMED

    def is_deleted(self):
        return self.state == TrackState.DELETED

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
# Distance metrics
# =============================================================================

def cosine_distance(a, b, data_is_normalized=True):
    """Compute cosine distance between feature vectors."""
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1.0 - np.dot(a, b.T)


def nn_cosine_distance(tracks, detections):
    """Compute nearest neighbor cosine distance for each track-detection pair."""
    cost_matrix = np.zeros((len(tracks), len(detections)))

    for i, track in enumerate(tracks):
        if not track.features:
            cost_matrix[i, :] = 1.0
            continue

        track_features = np.array(track.features)
        for j, det in enumerate(detections):
            if not det.features:
                cost_matrix[i, j] = 1.0
            else:
                det_feature = np.array(det.features[0]).reshape(1, -1)
                distances = cosine_distance(track_features, det_feature)
                cost_matrix[i, j] = distances.min()

    return cost_matrix


def iou_batch(bboxes1, bboxes2):
    """Compute IOU between two sets of bounding boxes."""
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


def iou_distance(tracks, detections):
    """Compute IOU distance (1 - IOU) matrix."""
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))

    tlbrs1 = np.array([t.tlbr for t in tracks])
    tlbrs2 = np.array([d.tlbr for d in detections])
    return 1.0 - iou_batch(tlbrs1, tlbrs2)


# =============================================================================
# Linear assignment with gating
# =============================================================================

INFTY_COST = 1e5


def linear_assignment(cost_matrix, thresh):
    """Perform linear assignment with threshold."""
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


def matching_cascade(distance_metric, max_distance, cascade_depth, tracks,
                     detections, track_indices=None, detection_indices=None):
    """Run matching cascade: match by age (recently seen first)."""
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []

    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:
            break

        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]

        if len(track_indices_l) == 0:
            continue

        cost_matrix = distance_metric(
            [tracks[i] for i in track_indices_l],
            [detections[i] for i in unmatched_detections]
        )

        matches_l, _, unmatched_detections_l = linear_assignment(
            cost_matrix, max_distance
        )

        for row, col in matches_l:
            matches.append((track_indices_l[row], unmatched_detections[col]))

        unmatched_detections = [unmatched_detections[i] for i in unmatched_detections_l]

    unmatched_tracks = [k for k in track_indices if k not in [m[0] for m in matches]]

    return matches, unmatched_tracks, unmatched_detections


# =============================================================================
# Converters to/from Kwiver types
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
                logger.warning("Unsorted input in to_ObjectTrackSet for track %d", track.track_id)
        result.append(t)
    return ObjectTrackSet(result)


# =============================================================================
# DeepSORT Configuration
# =============================================================================

class DeepSORTTrackerConfig(scfg.DataConfig):
    """Configuration for DeepSORT tracker."""
    max_dist = scfg.Value(0.2, help='Maximum cosine distance for appearance matching')
    min_confidence = scfg.Value(0.3, help='Minimum detection confidence threshold')
    max_iou_distance = scfg.Value(0.7, help='Maximum IOU distance for fallback matching')
    max_age = scfg.Value(30, help='Maximum frames to keep lost tracks')
    n_init = scfg.Value(3, help='Consecutive detections before track is confirmed')
    model_path = scfg.Value('', help='Path to Re-ID model weights (optional)')
    device = scfg.Value('cuda', help='Device for feature extraction (cuda or cpu)')


# =============================================================================
# DeepSORT Algorithm (TrackObjects implementation)
# =============================================================================

class DeepSORTTracker(TrackObjects):
    """
    DeepSORT multi-object tracker.

    Uses cascade matching with deep appearance features for robust
    re-identification after occlusions.

    Reference: Wojke et al., "Simple Online and Realtime Tracking with
    a Deep Association Metric" (ICIP 2017)
    """

    def __init__(self):
        TrackObjects.__init__(self)
        self._config = DeepSORTTrackerConfig()

        # Internal state
        self._kalman_filter = None
        self._feature_extractor = None
        self._tracks = []
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
        self._max_dist = float(self._config.max_dist)
        self._min_confidence = float(self._config.min_confidence)
        self._max_iou_distance = float(self._config.max_iou_distance)
        self._max_age = int(self._config.max_age)
        self._n_init = int(self._config.n_init)
        self._model_path = self._config.model_path
        self._device = self._config.device

        # Initialize tracker state
        self._kalman_filter = KalmanFilter()
        self._feature_extractor = FeatureExtractor(
            model_path=self._model_path if self._model_path else None,
            device=self._device
        )
        self._tracks = []
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
            The input image for the current frame.
        detections : DetectedObjectSet
            Detected objects from the current frame.

        Returns
        -------
        ObjectTrackSet
            Updated object track set containing all active tracks.
        """
        self._frame_id += 1

        # Get image as numpy array for feature extraction
        img_np = None
        if image is not None:
            try:
                img_np = image.image().asarray().astype('uint8')
            except:
                img_np = None

        # Convert detections
        det_list = to_DetectedObject_list(detections) if detections else []
        all_detections = []

        boxes = []
        for do in det_list:
            score = get_DetectedObject_score(do)
            if score < self._min_confidence:
                continue
            tlwh = get_DetectedObject_bbox_tlwh(do)
            tlbr = get_DetectedObject_bbox_tlbr(do)
            boxes.append(tlbr)
            all_detections.append(STrack(tlwh, score, detected_object=do))

        # Extract appearance features
        if img_np is not None and len(boxes) > 0:
            features = self._feature_extractor.extract(img_np, boxes)
            for det, feat in zip(all_detections, features):
                det.features = [feat]

        # Predict existing tracks
        for track in self._tracks:
            track.predict()

        # Split tracks by state
        confirmed_tracks = [t for t in self._tracks if t.is_confirmed()]
        unconfirmed_tracks = [t for t in self._tracks if t.is_tentative()]

        # === Cascade matching for confirmed tracks ===
        matches_a, unmatched_tracks_a, unmatched_detections = matching_cascade(
            nn_cosine_distance, self._max_dist, self._max_age,
            confirmed_tracks, all_detections
        )

        # === IOU matching for remaining ===
        iou_track_candidates = [
            confirmed_tracks[i] for i in unmatched_tracks_a
            if confirmed_tracks[i].time_since_update == 1
        ]
        iou_track_candidates += unconfirmed_tracks

        unmatched_tracks_a = [
            i for i in unmatched_tracks_a
            if confirmed_tracks[i].time_since_update != 1
        ]

        matches_b = []
        if len(iou_track_candidates) > 0 and len(unmatched_detections) > 0:
            cost_matrix = iou_distance(
                iou_track_candidates,
                [all_detections[i] for i in unmatched_detections]
            )
            matches_b, unmatched_tracks_b, unmatched_detections_b = linear_assignment(
                cost_matrix, self._max_iou_distance
            )

            for row, col in matches_b:
                track = iou_track_candidates[row]
                det = all_detections[unmatched_detections[col]]
                track.update(det, self._frame_id, ts)

            unmatched_detections = [unmatched_detections[i] for i in unmatched_detections_b]

        # Update matched tracks from cascade
        for track_idx, det_idx in matches_a:
            confirmed_tracks[track_idx].update(all_detections[det_idx], self._frame_id, ts)

        # Mark missed tracks
        for i in unmatched_tracks_a:
            confirmed_tracks[i].mark_missed()

        matched_iou_indices = set(row for row, _ in matches_b)
        for i, track in enumerate(iou_track_candidates):
            if i not in matched_iou_indices:
                track.mark_missed()

        # Initialize new tracks
        for det_idx in unmatched_detections:
            det = all_detections[det_idx]
            det._n_init = self._n_init
            det._max_age = self._max_age
            det.activate(self._kalman_filter, self._frame_id, ts)
            self._tracks.append(det)

        # Remove deleted tracks
        self._tracks = [t for t in self._tracks if not t.is_deleted()]

        # Output confirmed tracks
        output_tracks = [t for t in self._tracks if t.is_confirmed() and len(t.history) > 0]
        return to_ObjectTrackSet(output_tracks)

    def initialize(self, ts, image, seed_detections):
        """Initialize the tracker for a new sequence."""
        self.reset()
        if seed_detections is not None and len(seed_detections) > 0:
            return self.track(ts, image, seed_detections)
        return ObjectTrackSet([])

    def finalize(self):
        """Finalize tracking and return all tracks."""
        output_tracks = [t for t in self._tracks if len(t.history) > 0]
        return to_ObjectTrackSet(output_tracks)

    def reset(self):
        """Reset the tracker state."""
        self._kalman_filter = KalmanFilter()
        self._tracks = []
        self._frame_id = 0
        STrack.reset_id()


# =============================================================================
# Algorithm Registration
# =============================================================================

def __vital_algorithm_register__():
    """Register the DeepSORT algorithm with KWIVER."""
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "deepsort"

    if algorithm_factory.has_algorithm_impl_name(
            DeepSORTTracker.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "DeepSORT multi-object tracker with deep appearance features",
        DeepSORTTracker
    )

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
