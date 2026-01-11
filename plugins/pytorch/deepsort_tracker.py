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
# Kalman Filter for motion prediction (same as ByteTrack)
# =============================================================================

class KalmanFilter:
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space:
        x, y, a, h, vx, vy, va, vh

    where (x, y) is the center of the bounding box, a is the aspect ratio,
    h is the height, and v* are the respective velocities.
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

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """
        Compute gating distance (Mahalanobis) between state and measurements.
        """
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
    """
    Deep appearance feature extractor for Re-ID.

    Uses a CNN to extract appearance embeddings from detection crops.
    """

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

            # Use GPU if available
            if self.device == 'cuda' and not torch.cuda.is_available():
                self.device = 'cpu'

            # Load model
            if self.model_path and self.model_path.strip():
                # Load custom Re-ID model
                self.model = torch.load(self.model_path, map_location=self.device)
            else:
                # Use pretrained ResNet18 as feature extractor
                self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
                # Remove classification layer, keep features
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
        """
        Extract appearance features for detections.

        Parameters
        ----------
        image : ndarray
            Full frame image (H, W, C).
        boxes : list
            List of bounding boxes in (x1, y1, x2, y2) format.

        Returns
        -------
        ndarray
            Feature matrix (N, feature_dim).
        """
        self._initialize()

        if self.model is None:
            # Return random features if no model
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
    """
    Represents a single tracked object with Kalman filter state and appearance.
    """
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

        # Appearance features
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = 3  # Frames to confirm track
        self._max_age = 30  # Max frames to keep lost track

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

        # Update appearance features (keep recent ones)
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
    """
    Compute cosine distance between feature vectors.
    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1.0 - np.dot(a, b.T)


def nn_cosine_distance(tracks, detections):
    """
    Compute nearest neighbor cosine distance for each track-detection pair.
    """
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

def gate_cost_matrix(kf, cost_matrix, tracks, detections, gated_cost=INFTY_COST,
                     only_position=False):
    """
    Apply gating based on Mahalanobis distance.
    """
    gating_dim = 2 if only_position else 4
    gating_threshold = scipy.stats.chi2.ppf(0.95, df=gating_dim)

    measurements = np.array([d.tlwh_to_xyah(d._tlwh) for d in detections])

    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position
        )
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost

    return cost_matrix


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
    """
    Run matching cascade: match by age (recently seen first).
    """
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

        # Compute cost matrix for this level
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
# DeepSORT core algorithm
# =============================================================================

@Transformer.decorate
def deepsort_core(
    max_dist=0.2,
    min_confidence=0.3,
    max_iou_distance=0.7,
    max_age=30,
    n_init=3,
    model_path=None,
):
    """
    DeepSORT algorithm as a generator-based Transformer.

    Parameters
    ----------
    max_dist : float
        Maximum cosine distance for appearance matching.
    min_confidence : float
        Minimum detection confidence.
    max_iou_distance : float
        Maximum IOU distance for IOU matching stage.
    max_age : int
        Maximum frames to keep lost tracks.
    n_init : int
        Number of consecutive detections before track is confirmed.
    model_path : str
        Path to Re-ID model weights (optional).
    """
    kalman_filter = KalmanFilter()
    feature_extractor = FeatureExtractor(model_path=model_path)
    tracks = []
    frame_id = 0

    output = None
    while True:
        dos, ts, image = yield output
        frame_id += 1

        # Convert detections
        det_list = to_DetectedObject_list(dos)
        detections = []

        # Extract bounding boxes for feature extraction
        boxes = []
        for do in det_list:
            score = get_DetectedObject_score(do)
            if score < min_confidence:
                continue
            tlwh = get_DetectedObject_bbox_tlwh(do)
            tlbr = get_DetectedObject_bbox_tlbr(do)
            boxes.append(tlbr)
            detections.append(STrack(tlwh, score, detected_object=do))

        # Extract appearance features
        if image is not None and len(boxes) > 0:
            features = feature_extractor.extract(image, boxes)
            for det, feat in zip(detections, features):
                det.features = [feat]

        # Predict existing tracks
        for track in tracks:
            track.predict()

        # Split tracks by state
        confirmed_tracks = [t for t in tracks if t.is_confirmed()]
        unconfirmed_tracks = [t for t in tracks if t.is_tentative()]

        # === Cascade matching for confirmed tracks ===
        matches_a, unmatched_tracks_a, unmatched_detections = matching_cascade(
            nn_cosine_distance, max_dist, max_age,
            confirmed_tracks, detections
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

        if len(iou_track_candidates) > 0 and len(unmatched_detections) > 0:
            cost_matrix = iou_distance(
                iou_track_candidates,
                [detections[i] for i in unmatched_detections]
            )
            matches_b, unmatched_tracks_b, unmatched_detections_b = linear_assignment(
                cost_matrix, max_iou_distance
            )

            for row, col in matches_b:
                track = iou_track_candidates[row]
                det = detections[unmatched_detections[col]]
                track.update(det, frame_id, ts)

            unmatched_detections = [unmatched_detections[i] for i in unmatched_detections_b]
        else:
            unmatched_tracks_b = list(range(len(iou_track_candidates)))

        # Update matched tracks from cascade
        for track_idx, det_idx in matches_a:
            confirmed_tracks[track_idx].update(detections[det_idx], frame_id, ts)

        # Mark missed tracks
        for i in unmatched_tracks_a:
            confirmed_tracks[i].mark_missed()

        for i, track in enumerate(iou_track_candidates):
            if i in [row for row, _ in matches_b if 'matches_b' in dir()]:
                continue
            track.mark_missed()

        # Initialize new tracks
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            det._n_init = n_init
            det._max_age = max_age
            det.activate(kalman_filter, frame_id, ts)
            tracks.append(det)

        # Remove deleted tracks
        tracks = [t for t in tracks if not t.is_deleted()]

        # Output confirmed tracks
        output_tracks = [t for t in tracks if t.is_confirmed() and len(t.history) > 0]
        output = to_ObjectTrackSet(output_tracks)


# =============================================================================
# Sprokit process
# =============================================================================

def add_declare_config(proc, name_key, default, description):
    proc.add_config_trait(name_key, name_key, default, description)
    proc.declare_config_using_trait(name_key)


class deepsort_tracker(KwiverProcess):
    """
    DeepSORT multi-object tracker sprokit process.

    Uses cascade matching with deep appearance features for robust
    re-identification after occlusions.
    """

    def __init__(self, config):
        KwiverProcess.__init__(self, config)

        add_declare_config(self, "max_dist", "0.2",
            "Maximum cosine distance for appearance matching")
        add_declare_config(self, "min_confidence", "0.3",
            "Minimum detection confidence threshold")
        add_declare_config(self, "max_iou_distance", "0.7",
            "Maximum IOU distance for fallback matching")
        add_declare_config(self, "max_age", "30",
            "Maximum frames to keep lost tracks")
        add_declare_config(self, "n_init", "3",
            "Consecutive detections before track is confirmed")
        add_declare_config(self, "model_path", "",
            "Path to Re-ID model weights (optional)")

        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        self.declare_input_port_using_trait('detected_object_set', required)
        self.declare_input_port_using_trait('timestamp', required)
        self.declare_input_port_using_trait('image', optional)

        self.declare_output_port_using_trait('object_track_set', optional)

    def _configure(self):
        self._tracker = deepsort_core(
            max_dist=float(self.config_value('max_dist')),
            min_confidence=float(self.config_value('min_confidence')),
            max_iou_distance=float(self.config_value('max_iou_distance')),
            max_age=int(self.config_value('max_age')),
            n_init=int(self.config_value('n_init')),
            model_path=self.config_value('model_path'),
        )
        self._base_configure()

    def _step(self):
        dos = self.grab_input_using_trait('detected_object_set')
        ts = self.grab_input_using_trait('timestamp')

        # Image is optional for appearance features
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

    module_name = 'python:viame.processes.pytorch.deepsort_tracker'

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process(
        'deepsort_tracker',
        'DeepSORT multi-object tracker with deep appearance features',
        deepsort_tracker,
    )

    process_factory.mark_process_module_as_loaded(module_name)
