# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Circle detection by the Hough gradient transform.

This was `library/object_detectors/hough_circle_detector.cxx`, and P7-T04
moved it here on cv2, judging `cv::HoughCircles` too large to reproduce for
one shipped pipeline. It has since been reproduced: `image_kernels.canny` and
`image_kernels.hough_circles` are identical to cv2 over 160 and 864
configurations, so this file is off cv2 entirely and the registered name, the
six config keys and their defaults are still the C++ ones.

Findings 2.45 to 2.47 record what that took. The short version is that the
algorithm was never the hard part: what decided agreement was the 8-bit
Gaussian's fixed-point kernel, the value the output is sorted by, and the fact
that the whole transform runs in single precision.

What the C++ did to an image before the transform is reproduced exactly:

* the bridge converted the vital image to a BGR `cv::Mat` and then took
  `COLOR_BGR2GRAY`, which is `0.299 R + 0.587 G + 0.114 B` on the *original*
  channel order -- the two swaps cancel, so this takes `COLOR_RGB2GRAY` on
  the array as it stands rather than swapping twice;
* a 9x9 Gaussian at sigma 2 in both directions, to keep the edge detector
  off the noise.

Each circle becomes a detection whose box is the centre plus and minus the
radius, with confidence 1 and the single class `circle` -- OpenCV returns no
score per circle, so there is nothing else to put there.
"""

import logging

import numpy as np

from viame import image_kernels

from viame.algo import ImageObjectDetector
from viame.types import (BoundingBoxD, DetectedObject,
                                DetectedObjectSet, DetectedObjectType)

logger = logging.getLogger(__name__)

# The blur the C++ applied before the transform, in the C++ order:
# `cv::GaussianBlur( src, dst, cv::Size( 9, 9 ), 2, 2 )`.
BLUR_SIZE = (9, 9)
BLUR_SIGMA = 2


def _config_double(value):
    """Spell a double the way `config_block` spelled the C++ one.

    It streamed through an `ostringstream` at the default precision, so 1.0
    is "1" and not "1.0". The difference does not change how any of these
    values parses, but `registry.json` recorded the C++ spellings and a
    config default is part of the contract those hold.
    """
    return "%g" % float(value)


class HoughCircleDetector(ImageObjectDetector):
    """Detect circles with `image_kernels.hough_circles`."""

    def __init__(self):
        ImageObjectDetector.__init__(self)

        # The C++ PARAM_DEFAULTs, with their C++ types: the four thresholds
        # are doubles and the two radii are ints, which matters because
        # cv2 rejects a float radius.
        self._dp = 1.0
        self._min_dist = 100.0
        self._param1 = 200.0
        self._param2 = 100.0
        self._min_radius = 0
        self._max_radius = 0

    # ------------------------------------------------------------------
    # Configuration

    def get_configuration(self):
        cfg = super(ImageObjectDetector, self).get_configuration()
        cfg.set_value("dp", _config_double(self._dp))
        cfg.set_value("min_dist", _config_double(self._min_dist))
        cfg.set_value("param1", _config_double(self._param1))
        cfg.set_value("param2", _config_double(self._param2))
        cfg.set_value("min_radius", str(int(self._min_radius)))
        cfg.set_value("max_radius", str(int(self._max_radius)))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        self._dp = float(cfg.get_value("dp"))
        self._min_dist = float(cfg.get_value("min_dist"))
        self._param1 = float(cfg.get_value("param1"))
        self._param2 = float(cfg.get_value("param2"))
        self._min_radius = int(float(cfg.get_value("min_radius")))
        self._max_radius = int(float(cfg.get_value("max_radius")))

    def check_configuration(self, cfg):
        # The C++ only warned about keys it did not know, which the config
        # merge above already handles by ignoring them; there is no value of
        # the six that it rejected.
        return True

    # ------------------------------------------------------------------
    # Detection

    def detect(self, image_data):
        detections = DetectedObjectSet()

        if image_data is None:
            return detections

        image = image_data.asarray()

        # `COLOR_BGR2GRAY` on the bridge's BGR mat and `COLOR_RGB2GRAY` on
        # the array are the same arithmetic; see the module docstring.
        if image.ndim == 3 and image.shape[2] >= 3:
            gray = image_kernels.to_gray(
                np.ascontiguousarray(image[:, :, :3]))
        elif image.ndim == 3:
            gray = image[:, :, 0]
        else:
            gray = image

        # `BLUR_SIZE` is square and `gaussian_blur` takes the one extent.
        gray = image_kernels.gaussian_blur(
            np.ascontiguousarray(gray), BLUR_SIZE[0], float(BLUR_SIGMA))

        circles = image_kernels.hough_circles(
            gray,
            dp=self._dp,
            min_dist=self._min_dist,
            canny_threshold=self._param1,
            acc_threshold=self._param2,
            min_radius=self._min_radius,
            max_radius=self._max_radius)

        if len(circles) == 0:
            logger.debug("Detected 0 objects.")
            return detections

        # An N by 3 of x, y, radius, in the order cv2 returned its Vec3f rows
        # and the C++ read them.
        logger.debug("Detected %d objects.", len(circles))

        for x, y, radius in circles:
            box = BoundingBoxD(float(x - radius), float(y - radius),
                               float(x + radius), float(y + radius))

            types = DetectedObjectType("circle", 1.0)
            detections.add(DetectedObject(box, 1.0, types))

        return detections


def __vital_algorithm_register__():
    from viame.utilities.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        HoughCircleDetector, "hough_circle", "Hough circle detector")
