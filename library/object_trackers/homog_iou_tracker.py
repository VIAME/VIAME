# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Image-registration IoU tracker for fixed targets.

Built for high-resolution, low-frame-rate imagery of small stationary ground
targets (e.g. aerial surveys), where the camera moves far more between frames
than any target does. Each frame is registered to the previous one with a
feature-based homography; live track boxes are mapped through it and matched
to the current detections purely by IoU (Hungarian assignment). There is no
motion model: a target's apparent motion is entirely the camera's.
"""

import logging

import cv2
import numpy as np
import scriptconfig as scfg

from viame.algo import TrackObjects
from viame.types import ObjectTrackSet, ObjectTrackState, Track

from viame.object_trackers.simple_homog_tracker import (
    ious, optimize_iou_based_assignment, transform_matrix_box,
)

logger = logging.getLogger(__name__)


class HomogIOUTrackerConfig(scfg.DataConfig):
    """Configuration for the homography IoU tracker."""
    min_iou = scfg.Value(0.2, help='Minimum IoU, in registered coordinates, for a track-detection match')
    max_lost = scfg.Value(2, help='Frames an unmatched track is carried forward before it is terminated')
    new_track_thresh = scfg.Value(0.0, help='Minimum detection confidence to start a new track')
    registration_scale = scfg.Value(0.5, help='Image downscale factor used for feature registration')
    max_features = scfg.Value(4000, help='Maximum number of features detected per frame')
    feature_type = scfg.Value('sift', help='Feature detector for registration: sift or orb')
    min_inliers = scfg.Value(15, help='Minimum RANSAC inliers for a valid registration, else identity is used')


def _to_gray_uint8(image):
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image[:, :, 0]
        else:
            image = image[:, :, :3].astype(np.float32).mean(axis=2)
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        lo, hi = np.percentile(image, (0.5, 99.5))
        if hi <= lo:
            hi = lo + 1.0
        image = np.clip((image - lo) * (255.0 / (hi - lo)), 0, 255)
    return np.ascontiguousarray(image, dtype=np.uint8)


def _box_matrix(do):
    bbox = do.bounding_box
    return np.array([[bbox.min_x(), bbox.max_x()],
                     [bbox.min_y(), bbox.max_y()]], dtype=np.float64)


class _Track:
    __slots__ = ('track_id', 'box', 'homog', 'lost', 'history')

    def __init__(self, track_id, box, ts, do):
        self.track_id = track_id
        self.box = box
        # Maps the frame of the last observation into the current frame
        self.homog = np.eye(3)
        self.lost = 0
        self.history = [(ts, do)]


class HomogIOUTracker(TrackObjects):
    """
    Registration-only IoU tracker for fixed targets under a moving camera.
    """

    def __init__(self):
        TrackObjects.__init__(self)
        self._config = HomogIOUTrackerConfig()
        self._apply_config()

    def get_configuration(self):
        cfg = super(TrackObjects, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        from viame.utilities.utils import vital_config_update

        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._config.keys():
            self._config[key] = str(cfg.get_value(key))

        self._apply_config()
        return True

    def _apply_config(self):
        self._min_iou = float(self._config.min_iou)
        self._max_lost = int(self._config.max_lost)
        self._new_track_thresh = float(self._config.new_track_thresh)
        self._registration_scale = float(self._config.registration_scale)
        self._max_features = int(self._config.max_features)
        self._feature_type = str(self._config.feature_type).lower()
        self._min_inliers = int(self._config.min_inliers)

        if self._feature_type == 'sift':
            self._detector = cv2.SIFT_create(nfeatures=self._max_features)
            self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif self._feature_type == 'orb':
            self._detector = cv2.ORB_create(nfeatures=self._max_features)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            raise ValueError("Unknown feature_type: " + self._feature_type)

        self.reset()

    def check_configuration(self, cfg):
        if not cfg.has_value('feature_type'):
            return True
        return str(cfg.get_value('feature_type')).lower() in ('sift', 'orb')

    def _features(self, image):
        gray = _to_gray_uint8(image.asarray())
        scale = self._registration_scale
        if 0 < scale != 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)
        return self._detector.detectAndCompute(gray, None)

    def _estimate(self, prev, features):
        (kp0, des0), (kp1, des1) = prev, features
        if des0 is None or des1 is None or len(kp0) < 2 or len(kp1) < 2:
            logger.warning("homog_iou: too few features for registration, "
                           "using identity")
            return None

        good = [m[0] for m in self._matcher.knnMatch(des0, des1, k=2)
                if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]

        homog, inliers = None, 0
        if len(good) >= 4:
            src = np.float32([kp0[m.queryIdx].pt for m in good])
            dst = np.float32([kp1[m.trainIdx].pt for m in good])
            homog, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            inliers = int(mask.sum()) if mask is not None else 0

        if homog is None or inliers < self._min_inliers:
            logger.warning("homog_iou: registration failed (%d inliers of %d "
                           "matches), using identity", inliers, len(good))
            return None

        scale = self._registration_scale
        if 0 < scale != 1.0:
            s = np.diag([scale, scale, 1.0])
            homog = np.linalg.inv(s) @ homog @ s
        return homog / homog[2, 2]

    def _register(self, image):
        """Homography mapping previous-frame into current-frame pixels."""
        if image is None:
            self._prev_features = None
            return np.eye(3)

        features = self._features(image)
        prev = self._prev_features

        homog = self._estimate(prev, features) if prev is not None else None

        if homog is None and prev is not None:
            self._failures += 1
            # Lost tracks are still in the last good frame's coordinates, so
            # the next frame registers against it rather than the bad one
            if self._failures <= self._max_lost:
                return np.eye(3)

        self._failures = 0
        self._prev_features = features
        return np.eye(3) if homog is None else homog

    def track(self, ts, image, detections):
        self._frame_id += 1
        homog = self._register(image)

        det_list = list(detections) if detections else []
        det_boxes = [_box_matrix(do) for do in det_list]

        matches = [None] * len(self._tracks)
        if self._tracks and det_list:
            predicted = []
            for t in self._tracks:
                t.homog = homog @ t.homog
                predicted.append(transform_matrix_box(t.homog, t.box))
            iou = ious(np.array(predicted)[:, None], np.array(det_boxes)[None])
            matches = optimize_iou_based_assignment(iou, self._min_iou)
        else:
            for t in self._tracks:
                t.homog = homog @ t.homog

        used = set()
        live = []
        for t, di in zip(self._tracks, matches):
            if di is not None:
                t.box = det_boxes[di]
                t.homog = np.eye(3)
                t.lost = 0
                t.history.append((ts, det_list[di]))
                used.add(di)
                live.append(t)
            else:
                t.lost += 1
                if t.lost <= self._max_lost:
                    live.append(t)

        for di, do in enumerate(det_list):
            if di in used or do.confidence < self._new_track_thresh:
                continue
            self._next_id += 1
            live.append(_Track(self._next_id, det_boxes[di], ts, do))

        self._tracks = live
        return self._to_track_set(self._tracks)

    @staticmethod
    def _to_track_set(tracks):
        result = []
        for track in tracks:
            t = Track(id=track.track_id)
            for ts, do in track.history:
                ots = ObjectTrackState(ts.get_frame(), ts.get_time_usec(), do)
                if not t.append(ots):
                    logger.warning("Unsorted input for track %d", track.track_id)
            result.append(t)
        return ObjectTrackSet(result)

    def initialize(self, ts, image, seed_detections):
        self.reset()
        if seed_detections is not None and len(seed_detections) > 0:
            return self.track(ts, image, seed_detections)
        return ObjectTrackSet([])

    def finalize(self):
        return self._to_track_set(self._tracks)

    def reset(self):
        self._tracks = []
        self._next_id = 0
        self._frame_id = 0
        self._prev_features = None
        self._failures = 0


def __vital_algorithm_register__():
    from viame.utilities.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        HomogIOUTracker,
        "homog_iou",
        "Fixed-target tracker matching boxes by IoU after frame-to-frame "
        "homography registration",
    )
