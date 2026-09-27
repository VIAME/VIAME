# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Stereo disparity and depth.

the `opencv` plugin's `compute_stereo_disparity.cxx` in python, per
`lite-removals.md` section 2.4.

**The shipped SGBM path is off cv2**: `image_kernels.stereo_sgbm` is identical
to `cv::StereoSGBM` over 149 configurations of all three aggregations, the
three-way one included, which is the mode this selects (2.65). Rectification
went to `viame.measurement.projection` in P7-T06.

Two branches still reach for cv2, each behind an import inside itself so the
default path does not:

* `algorithm=BM`, because block matching is a different algorithm rather than a
  setting of this one and no shipped config selects it;
* `use_wls_filter`, which three shipped configs *do* set.
  `image_kernels.filter_disparity_wls` reproduces it closely and not exactly,
  and 2.67 says why that cannot be fixed -- `fastGlobalSmootherFilter` is not a
  function of its input, since cv2's own answer moves with `getNumThreads()`.
  The residue is about one pixel in six thousand, where we emit a small
  disparity and cv2 discards the pixel; the three `wls` golden variants are
  recorded from the C++ reference build at **zero** tolerance, so swapping this
  over is a decision about a measurement product rather than a rounding, and it
  is left to be taken deliberately.

The registered name, the sixteen config keys and their defaults are the C++
ones, and `tests/golden/measurement` holds this to what that produced on
eight of them.

Two conversions are reproduced rather than tidied, because the recording is
of their results:

* the C++ built a **BGR** `cv::Mat` and took `COLOR_BGR2GRAY` on it, which is
  the correct luminance of the original RGB, so this takes `COLOR_RGB2GRAY`
  on the array as it stands -- the two swaps cancel;
* a single channel result goes back as a plain image and a four channel one
  as colour, which the bridge wrote out as BGRA and so is RGBA here.
"""

import logging

import numpy as np

from viame import image_kernels
from viame.algo import ComputeStereoDepthMap
from viame.types import Image, ImageContainer

logger = logging.getLogger(__name__)

# `cv::StereoSGBM::create`'s arguments beyond the ones the config carries,
# in the order the C++ passed them: disp12MaxDiff, preFilterCap,
# uniquenessRatio. `MODE_SGBM_3WAY` is the mode, and it is not the default --
# it is faster than the full five-direction pass and gives a different map,
# so it has to be set rather than left out.
SGBM_DISP12_MAX_DIFF = 1
SGBM_PRE_FILTER_CAP = 0
SGBM_UNIQUENESS_RATIO = 10

# The penalties, as multiples of the block area. OpenCV's own documentation
# suggests these two numbers and the C++ used them.
SGBM_P1_FACTOR = 8
SGBM_P2_FACTOR = 32

# A matcher returns disparity in sixteenths of a pixel.
DISPARITY_SCALE = 16.0


def _as_bool(value):
    return str(value).strip().lower() in ("true", "yes", "on", "1")


def _config_bool(value):
    return "true" if value else "false"


def _config_double(value):
    """Spell a double as `config_block` spelled the C++ one: 1.5, not 1.5."""
    return "%g" % float(value)


def _to_gray(array):
    """The grey image the matcher sees.

    `calibrate_stereo_cameras::to_grayscale` on the bridge's BGR mat: one
    channel passes through, three take `BGR2GRAY` and four `BGRA2GRAY`. On
    the RGB array those are `RGB2GRAY` and `RGBA2GRAY`, which weight the
    channels the same way round.
    """
    from viame import image_kernels

    if array.ndim == 2:
        return array

    if array.shape[2] == 1:
        return array[:, :, 0]

    if array.shape[2] == 3:
        return image_kernels.to_gray(array)

    if array.shape[2] == 4:
        # The alpha plane is not part of the luminance; drop it and convert
        return image_kernels.to_gray(np.ascontiguousarray(array[:, :, :3]))

    return array[:, :, 0]


class ComputeStereoDisparity(ComputeStereoDepthMap):
    """Disparity or depth from a stereo pair, with BM or SGBM."""

    def __init__(self):
        ComputeStereoDepthMap.__init__(self)

        self._algorithm = "SGBM"
        self._min_disparity = 0
        self._num_disparities = 128
        self._sad_window_size = 21
        self._block_size = 5
        self._speckle_window_size = 100
        self._speckle_range = 32
        self._use_wls_filter = False
        self._wls_lambda = 8000.0
        self._wls_sigma = 1.5
        self._calibration_file = ""
        self._compute_depth = False
        self._output_format = "raw"
        self._uint16_scale_factor = 256.0
        self._output_rectified = True
        self._export_as_alpha = False

        self._calibration = None
        self._rectification = None

    # ------------------------------------------------------------------
    # Configuration

    def get_configuration(self):
        cfg = super(ComputeStereoDepthMap, self).get_configuration()
        cfg.set_value("algorithm", self._algorithm)
        cfg.set_value("min_disparity", str(int(self._min_disparity)))
        cfg.set_value("num_disparities", str(int(self._num_disparities)))
        cfg.set_value("sad_window_size", str(int(self._sad_window_size)))
        cfg.set_value("block_size", str(int(self._block_size)))
        cfg.set_value("speckle_window_size",
                      str(int(self._speckle_window_size)))
        cfg.set_value("speckle_range", str(int(self._speckle_range)))
        cfg.set_value("use_wls_filter", _config_bool(self._use_wls_filter))
        cfg.set_value("wls_lambda", _config_double(self._wls_lambda))
        cfg.set_value("wls_sigma", _config_double(self._wls_sigma))
        cfg.set_value("calibration_file", self._calibration_file)
        cfg.set_value("compute_depth", _config_bool(self._compute_depth))
        cfg.set_value("output_format", self._output_format)
        cfg.set_value("uint16_scale_factor",
                      _config_double(self._uint16_scale_factor))
        cfg.set_value("output_rectified", _config_bool(self._output_rectified))
        cfg.set_value("export_as_alpha", _config_bool(self._export_as_alpha))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        self._algorithm = str(cfg.get_value("algorithm"))
        self._min_disparity = int(float(cfg.get_value("min_disparity")))
        self._num_disparities = int(float(cfg.get_value("num_disparities")))
        self._sad_window_size = int(float(cfg.get_value("sad_window_size")))
        self._block_size = int(float(cfg.get_value("block_size")))
        self._speckle_window_size = int(
            float(cfg.get_value("speckle_window_size")))
        self._speckle_range = int(float(cfg.get_value("speckle_range")))
        self._use_wls_filter = _as_bool(cfg.get_value("use_wls_filter"))
        self._wls_lambda = float(cfg.get_value("wls_lambda"))
        self._wls_sigma = float(cfg.get_value("wls_sigma"))
        self._calibration_file = str(cfg.get_value("calibration_file"))
        self._compute_depth = _as_bool(cfg.get_value("compute_depth"))
        self._output_format = str(cfg.get_value("output_format"))
        self._uint16_scale_factor = float(
            cfg.get_value("uint16_scale_factor"))
        self._output_rectified = _as_bool(cfg.get_value("output_rectified"))
        self._export_as_alpha = _as_bool(cfg.get_value("export_as_alpha"))

        self._calibration = None
        self._rectification = None

    def check_configuration(self, cfg):
        algorithm = str(cfg.get_value("algorithm", self._algorithm))
        if algorithm not in ("BM", "SGBM"):
            logger.error("Invalid algorithm: %s. Must be 'BM' or 'SGBM'.",
                         algorithm)
            return False

        output_format = str(cfg.get_value("output_format",
                                          self._output_format))
        if output_format not in ("raw", "float32", "uint16_scaled"):
            logger.error(
                "Invalid output_format: %s. Must be 'raw', 'float32', or "
                "'uint16_scaled'.", output_format)
            return False

        return True

    # ------------------------------------------------------------------
    # The matchers

    def _sgbm_settings(self):
        """`cv::StereoSGBM_create`'s arguments, which are the C++'s.

        P1 and P2 scale with the block area -- 8 and 32 times it -- which is
        OpenCV's own documented suggestion and what the C++ passed.
        """
        block = self._block_size

        return dict(min_disparity=self._min_disparity,
                    num_disparities=self._num_disparities,
                    block_size=block,
                    p1=SGBM_P1_FACTOR * block * block,
                    p2=SGBM_P2_FACTOR * block * block,
                    disp12_max_diff=SGBM_DISP12_MAX_DIFF,
                    pre_filter_cap=SGBM_PRE_FILTER_CAP,
                    uniqueness_ratio=SGBM_UNIQUENESS_RATIO,
                    speckle_window_size=self._speckle_window_size,
                    speckle_range=self._speckle_range,
                    mode="sgbm_3way")

    def _compute_raw(self, left_rect, right_rect):
        """The disparity map in sixteenths, before any filtering.

        SGBM is `image_kernels.stereo_sgbm`, which is identical to cv2 over 149
        configurations of all three aggregations -- see lite-findings.md 2.65.
        **`algorithm=BM` is still cv2's**: block matching is a different
        algorithm rather than a setting of this one, no shipped config selects
        it, and the `bm` golden variant holds it to cv2's output exactly. The
        import is inside that branch so the SGBM path needs no cv2.
        """
        if self._algorithm == "BM":
            return self._cv_matcher().compute(left_rect, right_rect)

        if self._algorithm != "SGBM":
            raise RuntimeError(
                "Invalid algorithm type: " + self._algorithm)

        return image_kernels.stereo_sgbm(left_rect, right_rect,
                                         **self._sgbm_settings())

    def _cv_matcher(self):
        """cv2's matcher for the configured algorithm.

        Only the two branches that still need cv2 call this -- `algorithm=BM`
        and the WLS filter, which needs a `cv::StereoMatcher` to derive its
        right-view matcher from.
        """
        import cv2

        if self._algorithm == "BM":
            matcher = cv2.StereoBM_create(self._num_disparities,
                                          self._sad_window_size)
            matcher.setMinDisparity(self._min_disparity)
            matcher.setSpeckleWindowSize(self._speckle_window_size)
            matcher.setSpeckleRange(self._speckle_range)

            return matcher

        if self._algorithm != "SGBM":
            raise RuntimeError(
                "Invalid algorithm type: " + self._algorithm)

        settings = self._sgbm_settings()

        return cv2.StereoSGBM_create(
            settings["min_disparity"], settings["num_disparities"],
            settings["block_size"], settings["p1"], settings["p2"],
            settings["disp12_max_diff"], settings["pre_filter_cap"],
            settings["uniqueness_ratio"], settings["speckle_window_size"],
            settings["speckle_range"], cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def _apply_wls(self, left_rect, right_rect):
        """The WLS filter over the two views, which is **still cv2's**.

        `image_kernels.filter_disparity_wls` and `smooth_globally` exist and
        reproduce this closely, but not exactly, and the reason is in
        lite-findings.md 2.67: `fastGlobalSmootherFilter` is not a function of
        its input -- cv2's own answer moves with `getNumThreads()` -- so the
        remaining gap cannot be closed. It lands on about one pixel in six
        thousand, where the smoothed confidence has gone to nothing and the two
        implementations disagree about the sign of a quantity near zero; there
        we emit a small disparity and cv2 discards the pixel.
        
        That is a decision about a measurement product, not a rounding, and the
        three `wls` golden variants are recorded bit for bit from the C++
        reference build at **zero** tolerance. So this branch stays on cv2 until
        someone accepts a stated tolerance for it, and the kernels sit ready
        beside it.
        """
        import cv2

        matcher = self._cv_matcher()

        try:
            wls = cv2.ximgproc.createDisparityWLSFilter(matcher)
        except AttributeError:
            raise RuntimeError(
                "use_wls_filter needs a cv2 built with ximgproc, and "
                "this one has no cv2.ximgproc")

        wls.setLambda(self._wls_lambda)
        wls.setSigmaColor(self._wls_sigma)
        right_matcher = cv2.ximgproc.createRightMatcher(matcher)

        # `createDisparityWLSFilter` mutates the left matcher -- uniqueness off,
        # `disp12MaxDiff` at a million, speckles off -- so the map it filters
        # has to come from the matcher *after* that, not from `_compute_raw`.
        left_raw = matcher.compute(left_rect, right_rect)
        right_raw = right_matcher.compute(right_rect, left_rect)

        return wls.filter(left_raw, left_rect, None, right_raw, None,
                          right_rect)

    # ------------------------------------------------------------------
    # Rectification

    def _load_calibration(self):
        """The rig named by `calibration_file`, or None if there is none."""
        if not self._calibration_file:
            return None

        if self._calibration is None:
            from viame.measurement import _measurement

            loaded = _measurement.load_stereo_calibration(
                self._calibration_file)

            # The binding hands every matrix back **flat**, because it takes
            # and returns `std::vector< double >` to stay clear of pybind11's
            # Eigen caster. cv2 wants them shaped, and passing a nine element
            # vector where a 3 by 3 belongs fails inside `stereoRectify` with
            # an assertion about `cvConvertScale`, nowhere near the cause.
            shapes = {
                "k_left": (3, 3), "k_right": (3, 3),
                "rotation": (3, 3),
                "dist_left": (1, -1), "dist_right": (1, -1),
                "translation": (3, 1),
            }

            self._calibration = {
                key: np.asarray(value, dtype=np.float64).reshape(
                    shapes.get(key, (-1,)))
                for key, value in loaded.items()
            }

        return self._calibration

    def _rectification_maps(self, shape):
        """The four remap arrays, and the projection matrices they came from.

        `calibrate_stereo_cameras::ensure_rectification` is `stereoRectify`
        with `CALIB_ZERO_DISPARITY` and alpha zero, and the unrectification
        map is `undistortPoints` over the whole pixel grid -- which is not
        the inverse of the rectification map but the forward one, so
        remapping the disparity through it puts each rectified value back
        where its pixel came from.
        """
        from viame.measurement import projection

        if self._rectification is not None:
            return self._rectification

        calibration = self._load_calibration()
        height, width = shape

        rectify = projection.stereo_rectify(
            calibration["k_left"], calibration["dist_left"],
            calibration["k_right"], calibration["dist_right"],
            width, height, calibration["rotation"],
            calibration["translation"], alpha=0.0)

        r1 = rectify["left_rotation"]
        r2 = rectify["right_rotation"]
        p1 = rectify["left_projection"]
        p2 = rectify["right_projection"]

        left_x, left_y = projection.rectification_maps(
            calibration["k_left"], calibration["dist_left"], r1, p1,
            width, height)
        right_x, right_y = projection.rectification_maps(
            calibration["k_right"], calibration["dist_right"], r2, p2,
            width, height)

        grid = np.stack(np.meshgrid(np.arange(width, dtype=np.float64),
                                    np.arange(height, dtype=np.float64)),
                        axis=-1).reshape(-1, 2)

        undistorted = projection.undistort_points(
            grid, calibration["k_left"], calibration["dist_left"], r1, p1)
        undistorted = undistorted.reshape(height, width, 2)

        self._rectification = {
            "left_x": left_x, "left_y": left_y,
            "right_x": right_x, "right_y": right_y,
            "unrectify_x": np.ascontiguousarray(
                undistorted[:, :, 0], dtype=np.float32),
            "unrectify_y": np.ascontiguousarray(
                undistorted[:, :, 1], dtype=np.float32),
            "p1": p1, "p2": p2,
        }

        return self._rectification

    # ------------------------------------------------------------------

    def compute(self, left_image, right_image):
        import cv2
        from viame import image_kernels

        if left_image is None or right_image is None:
            logger.warning("Null input image(s)")
            return None

        left = left_image.asarray()
        right = right_image.asarray()

        if left.shape[:2] != right.shape[:2]:
            logger.warning("Inconsistent left/right image sizes")
            return None

        left_gray = _to_gray(left)
        right_gray = _to_gray(right)

        rectifying = bool(self._calibration_file)
        maps = self._rectification_maps(left_gray.shape[:2]) \
            if rectifying else None

        left_colour_rectified = None

        if rectifying:
            left_rect = image_kernels.remap(left_gray, maps["left_x"],
                                            maps["left_y"], "bilinear")
            right_rect = image_kernels.remap(right_gray, maps["right_x"],
                                             maps["right_y"], "bilinear")

            if self._export_as_alpha:
                left_colour_rectified = image_kernels.remap(
                    left, maps["left_x"], maps["left_y"], "bilinear")
        else:
            left_rect = left_gray
            right_rect = right_gray

            if self._export_as_alpha:
                left_colour_rectified = left

        if self._use_wls_filter:
            raw = self._apply_wls(left_rect, right_rect)
        else:
            raw = self._compute_raw(left_rect, right_rect)

        # Sixteenths of a pixel to pixels, and an unmatched pixel -- which a
        # matcher marks with a negative value -- to zero.
        float_map = raw.astype(np.float32) / DISPARITY_SCALE
        float_map[float_map < 0] = 0.0

        if self._compute_depth:
            if maps is None:
                raise RuntimeError(
                    "Cannot compute depth: calibration data missing.")

            focal = maps["p1"][0, 0]
            baseline = -maps["p2"][0, 3] / maps["p2"][0, 0]
            scale = focal * baseline

            depth = np.zeros_like(float_map)
            valid = float_map > 0
            depth[valid] = scale / float_map[valid]
            float_map = depth

        if rectifying and not self._output_rectified:
            # Nearest, so a depth is carried rather than blended with its
            # neighbours; float32 throughout, which the kernel takes as
            # such rather than truncating it to a byte.
            aligned = image_kernels.remap(
                float_map, maps["unrectify_x"], maps["unrectify_y"],
                "nearest", "constant", 0.0)
        else:
            aligned = float_map

        if self._output_format == "float32":
            formatted = aligned
        elif self._output_format == "uint16_scaled":
            formatted = np.clip(
                np.rint(aligned.astype(np.float64) *
                        self._uint16_scale_factor), 0, 65535).astype(np.uint16)
        elif self._compute_depth:
            # "raw" with depth is the metric depth, unscaled: there is no
            # sixteenths convention to go back to.
            formatted = aligned
        else:
            formatted = np.rint(
                aligned.astype(np.float64) * DISPARITY_SCALE).astype(np.int16)

        if not self._export_as_alpha:
            return ImageContainer(Image(np.ascontiguousarray(formatted)))

        colour = (left_colour_rectified
                  if rectifying and self._output_rectified
                  else left)

        if formatted.dtype == np.float32:
            converted = colour.astype(np.float32) / 255.0
        elif formatted.dtype == np.uint16:
            converted = np.clip(colour.astype(np.float64) * 257.0,
                                0, 65535).astype(np.uint16)
        else:
            converted = colour.astype(formatted.dtype)

        if converted.ndim == 2:
            converted = np.dstack([converted] * 3)

        output = np.dstack([converted[:, :, :3], formatted])

        return ImageContainer(Image(np.ascontiguousarray(output)))


def __vital_algorithm_register__():
    from viame.utilities.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        ComputeStereoDisparity, "ocv_stereo_disparity",
        "OpenCV stereo disparity map computation using BM or SGBM")
