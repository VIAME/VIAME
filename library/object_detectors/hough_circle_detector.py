# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Circle detection by the Hough gradient transform, on cv2.

This was `library/object_detectors/hough_circle_detector.cxx`, and P7-T04
moves it here rather than reimplementing it in `image_ops`, per
`lite-removals.md` section 2.6. `cv::HoughCircles` is not an imgproc
primitive of the kind `image_ops` carries: it is Canny plus a Sobel-gradient
accumulator plus a radius vote plus a non-maximum suppression over centres,
with tie-breaking rules that decide which of several nearby circles survives.
Reproducing it in C++ would be several hundred lines whose only justification
is one shipped pipeline, `detector_simple_hough.pipe`.

So the algorithm stays OpenCV's and only the language changes, which is what
the plan asks for wherever cv2 is allowed in python. The registered name,
the six config keys and their defaults are the C++ ones, so a pipeline that
selected `hough_circle` before selects this now with no edit.

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

from kwiver.vital.algo import ImageObjectDetector
from kwiver.vital.types import (BoundingBoxD, DetectedObject,
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
    """Detect circles with `cv2.HoughCircles`."""

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
        import cv2

        detections = DetectedObjectSet()

        if image_data is None:
            return detections

        image = image_data.asarray()

        # `COLOR_BGR2GRAY` on the bridge's BGR mat and `COLOR_RGB2GRAY` on
        # the array are the same arithmetic; see the module docstring.
        if image.ndim == 3 and image.shape[2] >= 3:
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
        elif image.ndim == 3:
            gray = image[:, :, 0]
        else:
            gray = image

        gray = cv2.GaussianBlur(gray, BLUR_SIZE, BLUR_SIGMA, BLUR_SIGMA)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            self._dp,
            self._min_dist,
            param1=self._param1,
            param2=self._param2,
            minRadius=self._min_radius,
            maxRadius=self._max_radius)

        if circles is None:
            logger.debug("Detected 0 objects.")
            return detections

        # cv2 returns (1, n, 3); the C++ got a vector of Vec3f from the same
        # call, so the rows are in the same order.
        circles = circles[0]
        logger.debug("Detected %d objects.", len(circles))

        for x, y, radius in circles:
            box = BoundingBoxD(float(x - radius), float(y - radius),
                               float(x + radius), float(y + radius))

            types = DetectedObjectType("circle", 1.0)
            detections.add(DetectedObject(box, 1.0, types))

        return detections


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        HoughCircleDetector, "hough_circle", "Hough circle detector")
