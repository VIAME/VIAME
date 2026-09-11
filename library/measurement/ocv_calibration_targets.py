# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Calibration target detection, on cv2.

`plugins/opencv/detect_calibration_targets.cxx` and the four detection
helpers it uses from `calibrate_stereo_cameras`, in python, per
`lite-removals.md` section 2.4. `findChessboardCorners`, `cornerSubPix` and
`SimpleBlobDetector` are calib3d and features2d rather than `image_ops`
primitives, so the algorithm stays OpenCV's.

Three things here are reproduced rather than corrected, because
`tests/golden/measurement` records what the C++ produced:

* the grey conversion has **red and blue swapped**. The C++ asked the bridge
  for an `RGB_COLOR` mat -- not the `BGR_COLOR` every other caller asks for
  -- and then took `COLOR_BGR2GRAY` on it, so the red and blue weights land
  the wrong way round. It makes no difference to a grey target and moves a
  corner by a fraction of a pixel on a coloured one, and changing it would
  change every calibration this has ever produced.
* `target_type` is compared against `checkerboard`, and any other value --
  including `chessboard`, the word the OpenCV function is named after --
  turns both detectors off and returns nothing.
* the dot path ignores `target_width` and `target_height` entirely; only the
  checkerboard path uses them.

`design/lite-findings.md` records all three.
"""

import logging
import math

import numpy as np

from kwiver.vital.algo import ImageObjectDetector
from kwiver.vital.types import (BoundingBoxD, DetectedObject,
                                DetectedObjectSet, DetectedObjectType)

logger = logging.getLogger(__name__)

# Every detection is a square this many pixels across, centred on the corner.
TARGET_WIDTH = 5

# Images larger than this are halved until they are not, and the detection is
# run on the smaller one and refined on the original.
MAX_DIMENSION = 5000

# `cornerSubPix`'s window and termination, from the C++.
SUBPIX_WINDOW = (11, 11)
SUBPIX_CRITERIA_ITERATIONS = 30
SUBPIX_CRITERIA_EPSILON = 0.001

# The grid sizes auto-detection tries first, in order, before its systematic
# sweep from MIN_GRID to MAX_GRID.
COMMON_GRIDS = ((6, 5), (7, 6), (8, 6), (9, 6), (5, 4), (8, 5), (7, 5))
MIN_GRID = 4
MAX_GRID = 15

# `SimpleBlobDetector::Params`, from the C++. The threshold sweep and the
# four shape filters; colour filtering is off because the image is inverted
# first.
BLOB_MIN_THRESHOLD = 40
BLOB_MAX_THRESHOLD = 220
BLOB_THRESHOLD_STEP = 10
BLOB_MIN_REPEATABILITY = 2
BLOB_MIN_CONVEXITY = 0.70
BLOB_MIN_INERTIA = 0.40

# Fewer than this many blobs is not a target.
MIN_DOTS = 3

# `refine_dot_centers`: the half-window, the contrast a window must have, and
# where in its range the threshold sits.
DOT_WINDOW_RADIUS = 7
DOT_MIN_CONTRAST = 10
DOT_THRESHOLD_FRACTION = 0.5

# `filter_target_cluster`: how many neighbours a dot needs within a radius of
# this many median nearest-neighbour distances, and the count below which no
# filtering happens at all.
CLUSTER_MIN_NEIGHBOURS = 3
CLUSTER_NEIGHBOUR_FACTOR = 3.0
CLUSTER_MIN_DOTS = 10


def _as_bool(value):
    return str(value).strip().lower() in ("true", "yes", "on", "1")


def _config_bool(value):
    return "true" if value else "false"


def _config_double(value):
    return "%g" % float(value)


def _to_gray(array):
    """The grey image the detectors see, red and blue swapped.

    See the module docstring: the C++ asked for an RGB mat and then took
    `BGR2GRAY` on it. Reproducing that means taking `BGR2GRAY` on the RGB
    array, which is the same wrong way round.
    """
    import cv2

    if array.ndim == 2:
        return array

    if array.shape[2] == 1:
        return array[:, :, 0]

    if array.shape[2] == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

    if array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)

    return array[:, :, 0]


def _detection_scale(shape):
    """The halving factor a large image is detected at."""
    scale = 1.0
    shortest = min(shape[0], shape[1])

    while scale * shortest > MAX_DIMENSION:
        scale /= 2.0

    return scale


def detect_chessboard(gray, grid):
    """`calibrate_stereo_cameras::detect_chessboard`.

    Returns `(found, corners, grid)`. A large image is detected and refined
    at the reduced size, scaled back up, and then refined again at full
    resolution -- both refinements, which is what the C++ does.
    """
    import cv2

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                SUBPIX_CRITERIA_ITERATIONS, SUBPIX_CRITERIA_EPSILON)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH

    scale = _detection_scale(gray.shape)

    if scale < 1.0:
        small = cv2.resize(gray, None, fx=scale, fy=scale)
        found, corners = cv2.findChessboardCorners(small, grid, flags=flags)

        if found:
            corners = cv2.cornerSubPix(small, corners, SUBPIX_WINDOW,
                                       (-1, -1), criteria)
            corners = corners / scale
    else:
        found, corners = cv2.findChessboardCorners(gray, grid, flags=flags)

    if not found:
        return False, [], grid

    corners = cv2.cornerSubPix(np.ascontiguousarray(gray),
                               np.ascontiguousarray(corners, dtype=np.float32),
                               SUBPIX_WINDOW, (-1, -1), criteria)

    return True, corners.reshape(-1, 2), grid


def _auto_grids():
    """Every grid auto-detection tries, in the order the C++ tries them."""
    grids = list(COMMON_GRIDS)

    for width in range(MIN_GRID, MAX_GRID + 1):
        for height in range(MIN_GRID, width + 1):
            # Already present in either orientation
            if any(grid == (width, height) or grid == (height, width)
                   for grid in grids):
                continue

            grids.append((width, height))

    return grids


def detect_chessboard_auto(gray):
    """`calibrate_stereo_cameras::detect_chessboard_auto`.

    Each candidate is tried, and then its transpose if it is not square, and
    the first that matches wins. Worth knowing what that means: a smaller
    grid inside a chessboard is also a chessboard, so this can settle on a
    sub-grid -- see `design/lite-findings.md`.
    """
    for grid in _auto_grids():
        found, corners, _ = detect_chessboard(gray, grid)

        if found:
            return True, corners, grid

        if grid[0] != grid[1]:
            transposed = (grid[1], grid[0])
            found, corners, _ = detect_chessboard(gray, transposed)

            if found:
                return True, corners, transposed

    logger.warning("Could not auto-detect chessboard grid size")
    return False, [], (0, 0)


def refine_dot_centers(gray, centers):
    """`calibrate_stereo_cameras::refine_dot_centers`.

    An intensity-weighted centroid over the bright half of a window's range.
    A window without enough contrast keeps the blob detector's own centre.
    """
    refined = []

    for centre in centers:
        cx = int(round(float(centre[0])))
        cy = int(round(float(centre[1])))

        x0 = max(0, cx - DOT_WINDOW_RADIUS)
        y0 = max(0, cy - DOT_WINDOW_RADIUS)
        x1 = min(gray.shape[1], cx + DOT_WINDOW_RADIUS + 1)
        y1 = min(gray.shape[0], cy + DOT_WINDOW_RADIUS + 1)

        if x1 <= x0 or y1 <= y0:
            refined.append(centre)
            continue

        window = gray[y0:y1, x0:x1].astype(np.float64)
        low = float(window.min())
        high = float(window.max())

        if high - low < DOT_MIN_CONTRAST:
            refined.append(centre)
            continue

        threshold = low + DOT_THRESHOLD_FRACTION * (high - low)
        bright = window >= threshold
        weights = np.where(bright, window - low, 0.0)
        total = float(weights.sum())

        if total <= 0.0:
            refined.append(centre)
            continue

        rows, columns = np.mgrid[y0:y1, x0:x1]
        refined.append((float((weights * columns).sum() / total),
                        float((weights * rows).sum() / total)))

    return refined


def detect_dots(gray, min_area, max_area, min_circularity):
    """`calibrate_stereo_cameras::detect_dots`.

    The image is inverted first, so the white dots become the dark blobs the
    detector looks for, and the area limits are scaled by the square of the
    detection scale along with it.
    """
    import cv2

    scale = _detection_scale(gray.shape)

    work = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1.0 else gray
    inverted = cv2.bitwise_not(work)

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = BLOB_MIN_THRESHOLD
    params.maxThreshold = BLOB_MAX_THRESHOLD
    params.thresholdStep = BLOB_THRESHOLD_STEP
    params.minRepeatability = BLOB_MIN_REPEATABILITY

    params.filterByArea = True
    params.minArea = min_area * scale * scale
    params.maxArea = max_area * scale * scale

    params.filterByCircularity = True
    params.minCircularity = min_circularity

    params.filterByConvexity = True
    params.minConvexity = BLOB_MIN_CONVEXITY

    params.filterByInertia = True
    params.minInertiaRatio = BLOB_MIN_INERTIA

    params.filterByColor = False

    keypoints = cv2.SimpleBlobDetector_create(params).detect(inverted)

    if len(keypoints) < MIN_DOTS:
        logger.debug("Dot detection: found only %d blobs (need >= %d)",
                     len(keypoints), MIN_DOTS)
        return False, [], (0, 0)

    centers = [(keypoint.pt[0] / scale, keypoint.pt[1] / scale)
               for keypoint in keypoints]
    centers = refine_dot_centers(gray, centers)

    return True, centers, (len(centers), 1)


def filter_target_cluster(centers):
    """`calibrate_stereo_cameras::filter_target_cluster`.

    Keeps the dots with enough neighbours within a few median
    nearest-neighbour distances, which drops a label or a reflection off to
    one side. Below ten dots nothing is filtered, and nothing is filtered if
    fewer than ten would survive.
    """
    count = len(centers)

    if count < CLUSTER_MIN_DOTS:
        return centers

    points = np.asarray(centers, dtype=np.float64)
    deltas = points[:, None, :] - points[None, :, :]
    distances = np.sqrt((deltas ** 2).sum(axis=-1))

    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)

    # The C++ sorts and takes element n/2, which for an even count is the
    # upper of the two middle values rather than their mean.
    median = float(np.sort(nearest)[count // 2])
    radius = CLUSTER_NEIGHBOUR_FACTOR * median

    neighbours = (distances <= radius).sum(axis=1)
    keep = neighbours >= CLUSTER_MIN_NEIGHBOURS

    if int(keep.sum()) < CLUSTER_MIN_DOTS:
        return centers

    return [centers[index] for index in range(count) if keep[index]]


def make_object_points(grid, square_size):
    """`calibrate_stereo_cameras::make_object_points`: the board in its own
    coordinates, row by row."""
    return [(column * square_size, row * square_size, 0.0)
            for row in range(grid[1])
            for column in range(grid[0])]


class DetectCalibrationTargets(ImageObjectDetector):
    """Find a chessboard's corners or a dot board's centres."""

    def __init__(self):
        ImageObjectDetector.__init__(self)

        self._config_file = ""
        self._target_type = "auto"
        self._square_size = 1.0
        self._auto_detect_grid = False
        self._target_width = 7
        self._target_height = 5
        self._object_type = "unknown"
        self._dot_min_area = 30.0
        self._dot_max_area = 5000.0
        self._dot_min_circularity = 0.65
        self._roi_x1 = -1
        self._roi_y1 = -1
        self._roi_x2 = -1
        self._roi_y2 = -1

        # Auto-detection latches onto the first grid it finds and reuses it
        # for every later frame, which is what the C++ does and why two
        # cameras can end up on different grids.
        self._detected_grid = None

    # ------------------------------------------------------------------
    # Configuration

    def get_configuration(self):
        cfg = super(ImageObjectDetector, self).get_configuration()
        cfg.set_value("config_file", self._config_file)
        cfg.set_value("target_type", self._target_type)
        cfg.set_value("square_size", _config_double(self._square_size))
        cfg.set_value("auto_detect_grid",
                      _config_bool(self._auto_detect_grid))
        cfg.set_value("target_width", str(int(self._target_width)))
        cfg.set_value("target_height", str(int(self._target_height)))
        cfg.set_value("object_type", self._object_type)
        cfg.set_value("dot_min_area", _config_double(self._dot_min_area))
        cfg.set_value("dot_max_area", _config_double(self._dot_max_area))
        cfg.set_value("dot_min_circularity",
                      _config_double(self._dot_min_circularity))
        cfg.set_value("roi_x1", str(int(self._roi_x1)))
        cfg.set_value("roi_y1", str(int(self._roi_y1)))
        cfg.set_value("roi_x2", str(int(self._roi_x2)))
        cfg.set_value("roi_y2", str(int(self._roi_y2)))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        self._config_file = str(cfg.get_value("config_file"))
        self._target_type = str(cfg.get_value("target_type"))
        self._square_size = float(cfg.get_value("square_size"))
        self._auto_detect_grid = _as_bool(cfg.get_value("auto_detect_grid"))
        self._target_width = int(float(cfg.get_value("target_width")))
        self._target_height = int(float(cfg.get_value("target_height")))
        self._object_type = str(cfg.get_value("object_type"))
        self._dot_min_area = float(cfg.get_value("dot_min_area"))
        self._dot_max_area = float(cfg.get_value("dot_max_area"))
        self._dot_min_circularity = float(
            cfg.get_value("dot_min_circularity"))
        self._roi_x1 = int(float(cfg.get_value("roi_x1")))
        self._roi_y1 = int(float(cfg.get_value("roi_y1")))
        self._roi_x2 = int(float(cfg.get_value("roi_x2")))
        self._roi_y2 = int(float(cfg.get_value("roi_y2")))

        self._detected_grid = None

    def check_configuration(self, cfg):
        return True

    # ------------------------------------------------------------------

    def _in_roi(self, point):
        return (self._roi_x1 <= point[0] <= self._roi_x2 and
                self._roi_y1 <= point[1] <= self._roi_y2)

    def detect(self, image_data):
        detections = DetectedObjectSet()

        if image_data is None:
            return detections

        gray = _to_gray(image_data.asarray())

        # `checkerboard`, not `chessboard`: any other value leaves both of
        # these false and the detector finds nothing.
        try_checkerboard = self._target_type in ("checkerboard", "auto")
        try_dots = self._target_type in ("dots", "auto")

        found = False
        corners = []
        grid = (self._target_width, self._target_height)
        dots = False

        if try_checkerboard:
            if self._auto_detect_grid and self._detected_grid is None:
                found, corners, grid = detect_chessboard_auto(gray)

                if found:
                    self._detected_grid = grid
                    logger.info("Auto-detected grid size: %dx%d", *grid)
            elif self._auto_detect_grid:
                grid = self._detected_grid
                found, corners, grid = detect_chessboard(gray, grid)
            else:
                found, corners, grid = detect_chessboard(gray, grid)

        if not found and try_dots:
            found, corners, grid = detect_dots(
                gray, self._dot_min_area, self._dot_max_area,
                self._dot_min_circularity)

            if found:
                corners = filter_target_cluster(corners)
                grid = (len(corners), 1)
                dots = True

        has_roi = (self._roi_x1 >= 0 and self._roi_y1 >= 0 and
                   self._roi_x2 > self._roi_x1 and
                   self._roi_y2 > self._roi_y1)

        if found and has_roi:
            corners = [point for point in corners if self._in_roi(point)]

            if not corners:
                found = False

        if not found:
            logger.warning("Unable to find an OCV target")
            return detections

        world = None

        if not dots:
            world = make_object_points(grid, self._square_size)

            if len(corners) != len(world):
                logger.warning(
                    "Corner count mismatch: detected %d, expected %d",
                    len(corners), len(world))
                return detections

        for index, point in enumerate(corners):
            box = BoundingBoxD(
                float(point[0]) - TARGET_WIDTH / 2.0,
                float(point[1]) - TARGET_WIDTH / 2.0,
                float(point[0]) - TARGET_WIDTH / 2.0 + TARGET_WIDTH,
                float(point[1]) - TARGET_WIDTH / 2.0 + TARGET_WIDTH)

            detection = DetectedObject(
                box, 1.0, DetectedObjectType(self._object_type, 1.0))

            if world is not None:
                # `std::to_string` on a float, which is six decimal places.
                for axis, value in zip("xyz", world[index]):
                    detection.add_note(
                        ":stereo3d_{}={:.6f}".format(axis, value))

            detections.add(detection)

        return detections


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        DetectCalibrationTargets, "ocv_detect_calibration_targets",
        "Detect calibration targets (checkerboard or dots) with OpenCV")
