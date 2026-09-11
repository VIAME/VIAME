# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Single camera calibration from a track set of target corners, on cv2.

`plugins/opencv/calibrate_single_camera_process.cxx` and the two utility
files behind it, in python. Same reason as its stereo counterpart: the fit
is `cv::calibrateCamera`, which is calib3d, so the numerics stay OpenCV's
and only the language changes.

`tests/golden/measurement`'s `mono_calibration` cases hold this to the
matrices the C++ wrote and to the rig the fixture views were rendered
through, the same pair of checks the stereo case gets.

Three behaviours are reproduced rather than corrected:

* the fit is **progressive**, as the stereo one is: the full model, then a
  fixed aspect ratio, then a fixed principal point, then each distortion
  coefficient in turn, keeping a constraint while it does not worsen the
  error past a quarter. The thresholds and the order are the C++'s. Unlike
  the stereo version the principal point test divides its horizontal offset
  by the width and its vertical by the **height**, which is the sensible
  thing and the opposite of what the stereo one does.
* the process **never sees the image and guesses its size** from how far the
  corners reach: the furthest corner, plus half a box, plus 100, rounded
  down to the next hundred. On a 640 by 480 frame that gives 700 by 400, and
  that pair is what the calibration file carries as the image size. Finding
  1.10 in `design/lite-findings.md` says what it costs.
* the attributes of a detection go into a `std::map` with `insert`, which
  **keeps the first** value for a name rather than the last.
"""

import logging
import os

import numpy as np

from kwiver.sprokit.pipeline import process
from kwiver.sprokit.processes.kwiver_process import KwiverProcess

logger = logging.getLogger(__name__)

# The progressive fit's thresholds, as `calibrate_stereo_cameras.cxx` has
# them. An aspect ratio within one per cent of square is fixed at one; a
# principal point within five per cent of the image of the centre is fixed
# there; and a distortion coefficient is fixed while doing so keeps the error
# within a quarter of what it was.
ASPECT_RATIO_TOLERANCE = 0.01
PRINCIPAL_POINT_TOLERANCE = 0.05
DISTORTION_ERROR_FACTOR = 1.25

# The initial guess. `cv::calibrateCamera` discards it without
# `CALIB_USE_INTRINSIC_GUESS`, which is never set here, so these matter only
# once the aspect ratio is fixed -- and by then they have been overwritten.
# Kept so the sequence of calls is the same one.
INITIAL_FOCAL_LENGTH = 1000.0

# What the world points are named in a detection's notes.
WORLD_KEYS = ("stereo3d_x", "stereo3d_y", "stereo3d_z")


def _parse_detection_attribute(note):
    """`parse_detection_attribute`: "(trk) :name=value" as (name, value).

    An unparseable note is ("", 0.0) rather than an error, which is what the
    C++ returns and what puts an empty key in the map.
    """
    colon = note.find(":")
    equals = note.find("=")

    if colon < 0 or equals < 0 or equals == colon + 1:
        return "", 0.0

    return note[colon + 1:equals], float(note[equals + 1:])


def _frame_points(tracks):
    """Every usable corner, as `{frame: ([image point], [world point])}`."""
    image_points = {}
    world_points = {}

    for track in tracks.tracks():
        for state in track:
            detection = state.detection()

            if not detection.notes:
                continue

            # `std::map::insert` keeps the first value for a name, not the
            # last, so a note that repeats one is ignored.
            attributes = {}
            for note in detection.notes:
                name, value = _parse_detection_attribute(note)
                attributes.setdefault(name, value)

            if not all(key in attributes for key in WORLD_KEYS):
                continue

            box = detection.bounding_box
            centre = ((box.min_x() + box.max_x()) / 2.0,
                      (box.min_y() + box.max_y()) / 2.0)

            frame = state.frame_id
            image_points.setdefault(frame, []).append(centre)
            world_points.setdefault(frame, []).append(
                tuple(attributes[key] for key in WORLD_KEYS))

    return image_points, world_points


def extract_calibration_data(tracks, square_size, frame_count_threshold):
    """`extract_calibration_data_from_tracks`.

    Returns `(image_points, object_points, grid)` or None when there is
    nothing usable. Both point lists are float32, which is what
    `cv::Point2f` and `cv::Point3f` made them.
    """
    image_points, world_points = _frame_points(tracks)

    if not image_points:
        logger.error("No valid calibration points found in tracks")
        return None

    # The grid comes from the lowest numbered frame's world points, which is
    # what iterating a `std::map` from its beginning gives.
    first = world_points[min(world_points)]
    max_x = max([point[0] for point in first] + [0.0])
    max_y = max([point[1] for point in first] + [0.0])

    grid = (int(np.float32(max_x) / np.float32(square_size)) + 1,
            int(np.float32(max_y) / np.float32(square_size)) + 1)

    logger.debug("Detected grid size: %dx%d", grid[0], grid[1])

    expected = grid[0] * grid[1]
    maximum = frame_count_threshold or len(image_points)

    kept_image = []
    kept_world = []

    for frame in sorted(image_points):
        if len(kept_image) >= maximum:
            break

        if len(image_points[frame]) != expected:
            continue

        kept_image.append(np.asarray(image_points[frame], dtype=np.float32))
        kept_world.append(np.asarray(world_points[frame], dtype=np.float32))

    logger.debug("Extracted %d valid frames for calibration", len(kept_image))

    if not kept_image:
        return None

    return kept_image, kept_world, grid


def estimate_image_size(tracks):
    """`estimate_image_size_from_tracks`, guess and all -- see finding 1.10."""
    width = 0
    height = 0

    for track in tracks.tracks():
        for state in track:
            box = state.detection().bounding_box
            width = max(width, int(box.max_x() + box.width() / 2))
            height = max(height, int(box.max_y() + box.height() / 2))

    return (((width + 100) // 100) * 100, ((height + 100) // 100) * 100)


def calibrate_single_camera(image_points, object_points, image_size):
    """The progressive fit. Returns `(intrinsics, distortion, rms)`."""
    import cv2

    intrinsics = np.eye(3, dtype=np.float64)
    intrinsics[0][0] = INITIAL_FOCAL_LENGTH
    intrinsics[1][1] = INITIAL_FOCAL_LENGTH
    intrinsics[0][2] = image_size[0] / 2.0
    intrinsics[1][2] = image_size[1] / 2.0

    distortion = np.zeros((5, 1), dtype=np.float64)
    flags = 0

    def fit(flags):
        return cv2.calibrateCamera(object_points, image_points, image_size,
                                   intrinsics, distortion, flags=flags)

    rms, intrinsics, distortion, _, _ = fit(flags)
    logger.debug("camera initial RMS: %s", rms)

    aspect = intrinsics[0][0] / intrinsics[1][1]

    if 1.0 - min(aspect, 1.0 / aspect) < ASPECT_RATIO_TOLERANCE:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
        rms, intrinsics, distortion, _, _ = fit(flags)

    # The horizontal offset over the width and the vertical over the height.
    offset = max(
        abs(intrinsics[0][2] - image_size[0] / 2.0) / image_size[0],
        abs(intrinsics[1][2] - image_size[1] / 2.0) / image_size[1])

    if offset < PRINCIPAL_POINT_TOLERANCE:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        rms, intrinsics, distortion, _, _ = fit(flags)

    threshold = DISTORTION_ERROR_FACTOR * rms

    for flag in (cv2.CALIB_ZERO_TANGENT_DIST, cv2.CALIB_FIX_K3,
                 cv2.CALIB_FIX_K2, cv2.CALIB_FIX_K1):
        previous = (intrinsics.copy(), distortion.copy(), flags, rms)

        flags |= flag
        error, intrinsics, distortion, _, _ = fit(flags)

        if error > threshold:
            intrinsics, distortion, flags, rms = previous
            break

        rms = error

    logger.debug("camera final RMS: %s", rms)

    return intrinsics, np.asarray(distortion).reshape(-1), rms


def _matrix(values):
    values = np.asarray(values, dtype=np.float64)
    values = values.reshape(values.shape[0], -1)
    return {"rows": int(values.shape[0]),
            "cols": int(values.shape[1]),
            "dt": "d",
            "data": [float(value) for value in values.reshape(-1)]}


def write_calibration(intrinsics, distortion, rms, output_directory,
                      json_filename, square_size, grid, image_size):
    """`write_mono_calibration`: `intrinsics.yml` beside a JSON summary."""
    from viame.file_io import _opencv_yaml

    directory = output_directory or "."

    _opencv_yaml.write(os.path.join(directory, "intrinsics.yml"), {
        "M1": _matrix(intrinsics),
        # `cv::Mat::zeros( 5, 1 )` went in, so one column comes out.
        "D1": _matrix(np.asarray(distortion).reshape(-1, 1)),
    })

    logger.debug("Wrote intrinsics to: %s/intrinsics.yml", directory)

    if not json_filename:
        return

    path = json_filename

    if not path.startswith("/") and ":/" not in path:
        path = directory + "/" + path

    coefficients = np.asarray(distortion).reshape(-1)

    def coefficient(index):
        return float(coefficients[index]) if index < len(coefficients) else 0.0

    # Hand-written rather than `json.dump`, because the C++ streamed each
    # number through `operator<<` at the default precision -- six significant
    # digits -- and a file written to more would not compare.
    def value(number):
        return "%g" % float(number)

    lines = ["{"]

    entries = [
        ("image_width", image_size[0]),
        ("image_height", image_size[1]),
        ("grid_width", grid[0]),
        ("grid_height", grid[1]),
        ("square_size_mm", square_size),
        ("rms_error", rms),
        ("fx", intrinsics[0][0]),
        ("fy", intrinsics[1][1]),
        ("cx", intrinsics[0][2]),
        ("cy", intrinsics[1][2]),
        ("k1", coefficient(0)),
        ("k2", coefficient(1)),
        ("p1", coefficient(2)),
        ("p2", coefficient(3)),
        ("k3", coefficient(4)),
    ]

    for index, (name, number) in enumerate(entries):
        comma = "," if index + 1 < len(entries) else ""
        lines.append('  "{}": {}{}'.format(name, value(number), comma))

    lines.append("}")

    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")

    logger.debug("Wrote JSON calibration to: %s", path)


class CalibrateSingleCamera(KwiverProcess):
    """`ocv_calibrate_single_camera`: one camera, from a corner track set."""

    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        for name, default, description in (
                ("output_directory", "./",
                 "Output directory for calibration files"),
                ("output_json_file", "calibration.json",
                 "Output JSON calibration file path"),
                ("frame_count_threshold", "50",
                 "Maximum number of frames to use during calibration. "
                 "0 to use all."),
                ("square_size", "80.0",
                 "Calibration pattern square size in world units (e.g., mm)")):
            self.add_config_trait(name, name, default, description)
            self.declare_config_using_trait(name)

        required = process.PortFlags()
        required.add(self.flag_required)

        self.add_port_trait("tracks", "object_track_set",
                            "Object track set with detected corners.")
        self.declare_input_port_using_trait("tracks", required)

    def _configure(self):
        self._output_directory = str(self.config_value("output_directory"))
        self._output_json_file = str(self.config_value("output_json_file"))
        self._frame_count_threshold = int(
            float(self.config_value("frame_count_threshold")))
        self._square_size = float(self.config_value("square_size"))

        self._base_configure()

    def _step(self):
        tracks = self.grab_input_using_trait("tracks")

        if tracks is None or not tracks.size():
            logger.warning("Received null object track set")
            self.mark_process_as_complete()
            return

        extracted = extract_calibration_data(
            tracks, self._square_size, self._frame_count_threshold)

        if extracted is None:
            logger.error("Failed to extract calibration data from tracks")
            self.mark_process_as_complete()
            return

        image_points, object_points, grid = extracted

        image_size = estimate_image_size(tracks)
        logger.debug("Estimated image size: %dx%d", *image_size)

        intrinsics, distortion, rms = calibrate_single_camera(
            image_points, object_points, image_size)

        logger.info("Calibration complete. RMS error: %s", rms)

        write_calibration(intrinsics, distortion, rms,
                          self._output_directory, self._output_json_file,
                          self._square_size, grid, image_size)

        self.mark_process_as_complete()


def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory

    module_name = "python:viame.measurement"

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process(
        "ocv_calibrate_single_camera",
        "Estimate one camera's intrinsics from a calibration target track set",
        CalibrateSingleCamera)

    process_factory.mark_process_module_as_loaded(module_name)
