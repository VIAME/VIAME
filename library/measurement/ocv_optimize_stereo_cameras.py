# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Stereo camera calibration, on cv2.

`plugins/opencv/optimize_stereo_cameras.cxx` in python, per
`lite-removals.md` section 2.4: `calibrateCamera`, `stereoCalibrate` and
`stereoRectify` are calib3d, so the numerics stay OpenCV's.

`tests/golden/measurement`'s `calibration_pipeline` cases hold this to the
matrices the C++ wrote **and** to the rig the fixture views were rendered
through, so a port that reproduces the recording and a port that is right are
the same thing here.

Two behaviours are reproduced rather than corrected, and both are in
`design/lite-findings.md`:

* the fit is **progressive** -- the full model, then fixing the aspect ratio,
  then the principal point, then each distortion coefficient in turn, keeping
  every constraint that does not worsen the error past a threshold. On clean
  data every constraint holds and the principal point ends up pinned at the
  image centre whatever it really was. Fitting the full model instead would
  disagree with every calibration file VIAME has written.
* `optimize` refuses when half the track count is odd, which means it refuses
  **any board with an odd number of corners** -- including the default 7 by 5
  target, which has thirty-five. The check it meant to make is that the
  tracks divide evenly between the two cameras. Kept because a pipeline that
  has been failing for a known reason should keep failing the same way until
  someone decides to fix it, rather than start producing calibrations that
  have never been looked at.
"""

import logging
import os

import numpy as np

from kwiver.vital.algo import OptimizeCameras
from kwiver.vital.types import (CameraMap, RotationD, SimpleCameraIntrinsics,
                                SimpleCameraPerspective, SimpleLandmarkMap,
                                FeatureTrackSet)

from viame.measurement import stereo_frame_selection as selection

logger = logging.getLogger(__name__)

# The distortion coefficients `calibrateCamera` is given room for. Seven, as
# the C++ allocated, which is k1 k2 p1 p2 k3 k4 k5 -- OpenCV reads the count
# to choose its model, and seven means the rational model is available.
DISTORTION_COEFFICIENTS = 7

# The progressive fit's thresholds. An aspect ratio within one per cent of
# square is fixed at one; a principal point within five per cent of the image
# width of the centre is fixed there; and a distortion coefficient is fixed
# when doing so keeps the error within a quarter of what it was.
ASPECT_RATIO_TOLERANCE = 0.01
PRINCIPAL_POINT_TOLERANCE = 0.05
DISTORTION_ERROR_FACTOR = 1.25


def _config_double(value):
    return "%g" % float(value)


def _rig_file_names(directory):
    root = directory if directory else "."
    return (os.path.join(root, "intrinsics.yml"),
            os.path.join(root, "extrinsics.yml"))


class OptimizeStereoCameras(OptimizeCameras):
    """Calibrate a stereo rig from the tracks of a calibration target."""

    def __init__(self):
        OptimizeCameras.__init__(self)

        self._image_width = 0
        self._image_height = 0
        self._frame_count_threshold = 0
        self._output_calibration_directory = ""
        self._output_json_file = ""
        self._square_size = 1.0

    # ------------------------------------------------------------------
    # Configuration

    def get_configuration(self):
        cfg = super(OptimizeCameras, self).get_configuration()
        cfg.set_value("image_width", str(int(self._image_width)))
        cfg.set_value("image_height", str(int(self._image_height)))
        cfg.set_value("frame_count_threshold",
                      str(int(self._frame_count_threshold)))
        cfg.set_value("output_calibration_directory",
                      self._output_calibration_directory)
        cfg.set_value("output_json_file", self._output_json_file)
        cfg.set_value("square_size", _config_double(self._square_size))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        self._image_width = int(float(cfg.get_value("image_width")))
        self._image_height = int(float(cfg.get_value("image_height")))
        self._frame_count_threshold = int(
            float(cfg.get_value("frame_count_threshold")))
        self._output_calibration_directory = str(
            cfg.get_value("output_calibration_directory"))
        self._output_json_file = str(cfg.get_value("output_json_file"))
        self._square_size = float(cfg.get_value("square_size"))

    def check_configuration(self, cfg):
        return True

    # ------------------------------------------------------------------
    # Writing what was found

    def _write_rig(self, k_left, k_right, dist_left, dist_right,
                   rotation, translation, r1, r2, p1, p2, q):
        """The two OpenCV YAML documents, through `library/file_io`."""
        from viame.file_io import _opencv_yaml

        intrinsics_file, extrinsics_file = _rig_file_names(
            self._output_calibration_directory)

        def matrix(values):
            values = np.asarray(values, dtype=np.float64)
            values = values.reshape(values.shape[0], -1)
            return {"rows": int(values.shape[0]),
                    "cols": int(values.shape[1]),
                    "dt": "d",
                    "data": [float(v) for v in values.reshape(-1)]}

        _opencv_yaml.write(intrinsics_file, {
            "M1": matrix(k_left),
            "M2": matrix(k_right),
            # `cv::Mat( D1 )` of a vector is one column, which is the shape
            # every calibration file in the tree holds.
            "D1": matrix(np.asarray(dist_left).reshape(-1, 1)),
            "D2": matrix(np.asarray(dist_right).reshape(-1, 1)),
        })

        _opencv_yaml.write(extrinsics_file, {
            "R": matrix(rotation),
            "T": matrix(np.asarray(translation).reshape(-1, 1)),
            "R1": matrix(r1),
            "R2": matrix(r2),
            "P1": matrix(p1),
            "P2": matrix(p2),
            "Q": matrix(q),
        })

    def _write_json(self, k_left, k_right, dist_left, dist_right,
                    rotation, translation, grid, rms):
        """`calibrate_stereo_cameras::write_calibration_json`.

        Hand-written rather than `json.dump`, because the C++ streamed each
        number through `operator<<` at the default precision -- six
        significant digits -- and a file written to more would not compare.
        """
        if not self._output_json_file:
            return

        def value(number):
            return "%g" % float(number)

        def coefficient(coefficients, index):
            coefficients = np.asarray(coefficients).reshape(-1)
            return (float(coefficients[index])
                    if index < len(coefficients) else 0.0)

        lines = ["{"]

        def entry(name, number, comma=True):
            lines.append('  "{}": {}{}'.format(name, value(number),
                                               "," if comma else ""))

        entry("image_width", self._image_width)
        entry("image_height", self._image_height)
        entry("grid_width", grid[0])
        entry("grid_height", grid[1])
        entry("square_size_mm", self._square_size)

        # The per-camera RMS is not carried here: the C++ built its JSON
        # result from the stereo pass alone and left both mono values at
        # their default zero.
        entry("rms_error_left", 0.0)
        entry("rms_error_right", 0.0)
        entry("rms_error_stereo", rms)

        for side, intrinsics, coefficients in (("left", k_left, dist_left),
                                               ("right", k_right, dist_right)):
            entry("fx_" + side, intrinsics[0][0])
            entry("fy_" + side, intrinsics[1][1])
            entry("cx_" + side, intrinsics[0][2])
            entry("cy_" + side, intrinsics[1][2])
            for index, name in enumerate(("k1", "k2", "p1", "p2", "k3")):
                entry("{}_{}".format(name, side),
                      coefficient(coefficients, index))

        translation = np.asarray(translation).reshape(-1)
        lines.append('  "T": [{}, {}, {}],'.format(
            value(translation[0]), value(translation[1]),
            value(translation[2])))

        rotation = np.asarray(rotation).reshape(3, 3)
        lines.append('  "R": [{}]'.format(
            ", ".join(value(v) for v in rotation.reshape(-1))))

        lines.append("}")

        with open(self._output_json_file, "w") as handle:
            handle.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # The fit

    def _points(self, feature_track_sets, landmark_maps):
        """The calibration point sets, or None if they are unusable."""
        points = selection.select_frames(feature_track_sets, landmark_maps,
                                         self._frame_count_threshold)

        if not points.is_usable():
            logger.warning("Unable to proceed with camera calibration.")
            return None

        logger.info("Calibration data prepared:")
        logger.info("  - Image size: %dx%d", self._image_width,
                    self._image_height)
        logger.info("  - World points: %d frames", len(points.world_pts))

        return points

    def _try_improve(self, world, image, intrinsics, distortion, flags,
                     max_error, context):
        """One `calibrateCamera`, kept only if it does not make things worse.

        Returns `(kept, intrinsics, distortion, error)`. Rolling back rather
        than keeping the better of the two is what the C++ does, and it is
        why the constraints compound in the order they are tried.
        """
        import cv2

        previous_intrinsics = intrinsics.copy()
        previous_distortion = distortion.copy()

        logger.info("  - Running intrinsic calibration: %s", context)

        error, intrinsics, distortion, _, _ = cv2.calibrateCamera(
            world, image, (self._image_width, self._image_height),
            intrinsics, distortion, flags=flags)

        logger.info("    Calibration error: %s", error)

        if error < max_error:
            return True, intrinsics, distortion, error

        logger.info("    Error too high, keeping previous parameters")
        return False, previous_intrinsics, previous_distortion, error

    def _calibrate_camera(self, world, image, name):
        """The progressive per-camera fit. Returns `(intrinsics, distortion)`."""
        import cv2

        intrinsics = cv2.initCameraMatrix2D(
            world, image, (self._image_width, self._image_height), 0)
        distortion = np.zeros((1, DISTORTION_COEFFICIENTS), dtype=np.float64)

        logger.info("Calibrating %s camera (%d frames)...", name, len(world))

        flags = 0
        unbounded = float(np.finfo(np.float64).max)

        _, intrinsics, distortion, error = self._try_improve(
            world, image, intrinsics, distortion, flags, unbounded, "Initial")

        aspect = intrinsics[0][0] / intrinsics[1][1]
        logger.info("  - Aspect ratio: %s", aspect)

        if 1.0 - min(aspect, 1.0 / aspect) < ASPECT_RATIO_TOLERANCE:
            flags |= cv2.CALIB_FIX_ASPECT_RATIO
            _, intrinsics, distortion, error = self._try_improve(
                world, image, intrinsics, distortion, flags, unbounded,
                "Fixing aspect ratio at 1.0")

        centre_x = intrinsics[0][2]
        centre_y = intrinsics[1][2]
        logger.info("  - Principal point: (%s, %s)", centre_x, centre_y)

        # Both offsets are divided by the image **width**, including the
        # vertical one. Reproduced: on a wide image it makes the vertical
        # test the looser of the two.
        offset = max(
            abs(centre_x - self._image_width / 2.0) / self._image_width,
            abs(centre_y - self._image_height / 2.0) / self._image_width)

        if offset < PRINCIPAL_POINT_TOLERANCE:
            flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
            _, intrinsics, distortion, error = self._try_improve(
                world, image, intrinsics, distortion, flags, unbounded,
                "Fixed principal point to image center")

        max_error = DISTORTION_ERROR_FACTOR * error

        for flag, context in ((cv2.CALIB_ZERO_TANGENT_DIST,
                               "No tangential distortion"),
                              (cv2.CALIB_FIX_K3, "No K3 distortion"),
                              (cv2.CALIB_FIX_K2, "No K2 distortion"),
                              (cv2.CALIB_FIX_K1, "No K1 distortion")):
            flags |= flag
            kept, intrinsics, distortion, error = self._try_improve(
                world, image, intrinsics, distortion, flags, max_error,
                context)

            if not kept:
                break

        return intrinsics, np.asarray(distortion).reshape(-1)

    def _calibrate_stereo(self, points, left, right):
        """`stereoCalibrate` with the intrinsics fixed, then rectification."""
        import cv2

        k_left, dist_left = left
        k_right, dist_right = right

        world = points.world_pts
        image_left = points.image_pts[0]
        image_right = points.image_pts[1]

        logger.info("Running stereo calibration...")

        rms, k_left, dist_left, k_right, dist_right, rotation, translation, \
            essential, fundamental = cv2.stereoCalibrate(
                world, image_left, image_right,
                k_left, dist_left, k_right, dist_right,
                (self._image_width, self._image_height),
                flags=cv2.CALIB_FIX_INTRINSIC)

        logger.info("Stereo calibration complete, RMS error: %s", rms)

        r1, r2, p1, p2, q = cv2.stereoRectify(
            k_left, dist_left, k_right, dist_right,
            (self._image_width, self._image_height), rotation, translation,
            flags=cv2.CALIB_ZERO_DISPARITY)[:5]

        logger.info("Computing stereo rectification...")
        logger.info("Writing calibration files...")

        self._write_rig(k_left, k_right, dist_left, dist_right,
                        rotation, translation, r1, r2, p1, p2, q)

        grid = self._grid_size(world)
        self._write_json(k_left, k_right, dist_left, dist_right,
                         rotation, translation, grid, rms)

        self._report_epipolar_error(points, k_left, dist_left,
                                    k_right, dist_right, fundamental)

        return (k_left, dist_left), (k_right, dist_right), rotation, translation

    def _grid_size(self, world):
        """The board's grid, from the extent of its world points."""
        if not len(world) or not len(world[0]):
            return (0, 0)

        first = np.asarray(world[0], dtype=np.float64)

        return (int(first[:, 0].max() / self._square_size) + 1,
                int(first[:, 1].max() / self._square_size) + 1)

    def _report_epipolar_error(self, points, k_left, dist_left,
                               k_right, dist_right, fundamental):
        """`m2^T F m1 = 0`, averaged, which is the quality check the C++ logs.

        Computed and logged rather than returned: nothing consumes it, and
        reproducing the log line is part of reproducing the implementation.
        """
        import cv2

        total = 0.0
        count = 0

        for index in range(len(points.image_pts[0])):
            left = points.image_pts[0][index].reshape(-1, 1, 2)
            right = points.image_pts[1][index].reshape(-1, 1, 2)

            left = cv2.undistortPoints(left, k_left, dist_left, P=k_left)
            right = cv2.undistortPoints(right, k_right, dist_right, P=k_right)

            left_lines = cv2.computeCorrespondEpilines(
                left, 1, fundamental).reshape(-1, 3)
            right_lines = cv2.computeCorrespondEpilines(
                right, 2, fundamental).reshape(-1, 3)

            original_left = points.image_pts[0][index]
            original_right = points.image_pts[1][index]

            for point in range(len(original_left)):
                total += abs(original_left[point][0] * right_lines[point][0] +
                             original_left[point][1] * right_lines[point][1] +
                             right_lines[point][2])
                total += abs(original_right[point][0] * left_lines[point][0] +
                             original_right[point][1] * left_lines[point][1] +
                             left_lines[point][2])

            count += len(original_left)

        if count:
            logger.info("Quality check - average epipolar error: %s",
                        total / count)

    # ------------------------------------------------------------------

    def optimize(self, cameras, tracks, landmarks, constraints=None):
        """Calibrate the pair. Returns the optimised camera map."""
        if cameras is None or tracks is None or landmarks is None:
            return cameras

        camera_dict = cameras.as_dict()

        if len(camera_dict) != 2:
            logger.warning("This optimizer only works for a stereo setup.")
            return cameras

        all_tracks = list(tracks.tracks())
        all_landmarks = landmarks.landmarks()

        half_tracks = len(all_tracks) // 2
        half_landmarks = len(all_landmarks) // 2

        # See the module docstring: this rejects an odd corner count, which
        # is the default target's.
        if half_tracks % 2 or half_landmarks % 2:
            logger.warning("Inconsistant features or landmarks number.")
            return cameras

        left_tracks = all_tracks[:half_tracks]
        right_tracks = all_tracks[half_tracks:]

        left_landmarks = {index: all_landmarks[index]
                          for index in range(half_tracks)
                          if index in all_landmarks}
        right_landmarks = {index - half_tracks: all_landmarks[index]
                           for index in range(half_tracks, len(all_tracks))
                           if index in all_landmarks}

        # The right camera's tracks are renumbered to match its landmarks,
        # which is what the C++ does with `set_id`. Rebuilt rather than
        # mutated, since a python Track's id is not settable.
        right_tracks = [_renumber(track, index)
                        for index, track in enumerate(right_tracks)]

        return self._optimize_pair(
            camera_dict,
            [FeatureTrackSet(left_tracks), FeatureTrackSet(right_tracks)],
            [SimpleLandmarkMap(left_landmarks),
             SimpleLandmarkMap(right_landmarks)])

    def _optimize_pair(self, camera_dict, feature_track_sets, landmark_maps):
        points = self._points(feature_track_sets, landmark_maps)

        if points is None:
            return CameraMap(camera_dict)

        left = self._calibrate_camera(points.world_pts, points.image_pts[0],
                                      "left")
        right = self._calibrate_camera(points.world_pts, points.image_pts[1],
                                       "right")

        left, right, rotation, translation = self._calibrate_stereo(
            points, left, right)

        keys = sorted(camera_dict)

        return CameraMap({
            keys[0]: _perspective_camera(left[0], left[1]),
            keys[1]: _perspective_camera(right[0], right[1],
                                         rotation, translation),
        })

    def optimize_camera(self, camera, features, landmarks, constraints=None):
        """The single-camera form, which the C++ left empty."""
        return camera


def _renumber(track, identifier):
    """A copy of `track` under a new id, states and all."""
    from kwiver.vital.types import Track

    renumbered = Track(identifier)

    for state in track:
        renumbered.append(state)

    return renumbered


def _perspective_camera(intrinsics, distortion, rotation=None,
                        translation=None):
    """A vital camera from an OpenCV intrinsic matrix and pose."""
    import cv2

    camera = SimpleCameraPerspective()
    camera.set_intrinsics(SimpleCameraIntrinsics(
        np.asarray(intrinsics, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64).reshape(-1)))

    if rotation is not None:
        vector, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))
        camera.set_rotation(RotationD(vector.reshape(-1)))
        camera.set_translation(
            np.asarray(translation, dtype=np.float64).reshape(-1))

    return camera


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        OptimizeStereoCameras, "ocv_optimize_stereo_cameras",
        "Camera optimizer for stereo configurations.")
