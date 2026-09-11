# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""RANSAC estimation of a homography and a fundamental matrix, on cv2.

`library/image_processing/estimate_{homography,fundamental_matrix}.cxx` in
python, per `lite-removals.md` section 2.4: `cv::findHomography` and
`cv::findFundamentalMat` are a minimal solver, a RANSAC loop and a
Levenberg-Marquardt refinement each, with sampling and tie-breaking rules that
decide which of several consistent models survives. They stay OpenCV's.

Both return `(matrix, inliers)`. That is the convention for a python
implementation of a method with an output parameter, and here it is the whole
point: every C++ caller of these uses the inlier flags rather than the matrix
alone -- `match_features_homography` keeps only the matches the estimator
called inliers -- so an implementation that could not return them would
quietly reduce every such matcher to producing nothing. See
`python/kwiver/vital/algo/trampolines/README.md`.
"""

import logging

import numpy as np

from kwiver.vital.algo import EstimateFundamentalMatrix, EstimateHomography
from kwiver.vital.types import FundamentalMatrixD, HomographyD

logger = logging.getLogger(__name__)

# The C++ refused below these and said so, rather than letting OpenCV assert.
MIN_HOMOGRAPHY_POINTS = 4
MIN_FUNDAMENTAL_POINTS = 8


def _points(values):
    """A vital point list as the `CV_32F` array `cv2` wants.

    Float32 because the C++ built `std::vector<cv::Point2f>`: the estimate is
    made from single-precision coordinates and would move slightly if this
    passed doubles.
    """
    return np.array([[float(p[0]), float(p[1])] for p in values],
                    dtype=np.float32)


def _flags(mask, count):
    """An OpenCV inlier mask as the list of bools the C++ built.

    `inliers.resize( mask.rows )` and then a byte per row: an empty mask --
    which is what OpenCV returns when it fails -- gives an empty list rather
    than a list of False, and that difference is visible to a caller that
    zips it against its matches.
    """
    if mask is None:
        return []

    return [bool(value) for value in np.asarray(mask).reshape(-1)[:count]]


class EstimateHomographyOCV(EstimateHomography):
    """Estimate a homography with `cv2.findHomography` and RANSAC."""

    def __init__(self):
        EstimateHomography.__init__(self)

    def get_configuration(self):
        # No keys: the C++ had none either.
        return super(EstimateHomography, self).get_configuration()

    def set_configuration(self, cfg_in):
        pass

    def check_configuration(self, cfg):
        return True

    def estimate(self, pts1, pts2, inlier_scale=1.0):
        import cv2

        if len(pts1) < MIN_HOMOGRAPHY_POINTS or \
                len(pts2) < MIN_HOMOGRAPHY_POINTS:
            logger.error("Not enough points to estimate a homography")
            return None, []

        matrix, mask = cv2.findHomography(
            _points(pts1), _points(pts2), cv2.RANSAC, inlier_scale)

        if matrix is None:
            return None, []

        return HomographyD(np.asarray(matrix, dtype=np.float64)), \
            _flags(mask, len(pts1))


class EstimateFundamentalMatrixOCV(EstimateFundamentalMatrix):
    """Estimate a fundamental matrix with `cv2.findFundamentalMat`."""

    def __init__(self):
        EstimateFundamentalMatrix.__init__(self)
        self._confidence_threshold = 0.99

    def get_configuration(self):
        cfg = super(EstimateFundamentalMatrix, self).get_configuration()
        cfg.set_value("confidence_threshold",
                      "%g" % float(self._confidence_threshold))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)
        self._confidence_threshold = float(
            cfg.get_value("confidence_threshold"))

    def check_configuration(self, cfg):
        threshold = float(cfg.get_value(
            "confidence_threshold", "%g" % self._confidence_threshold))

        if threshold <= 0.0 or threshold > 1.0:
            logger.error(
                "confidence_threshold parameter is %s, needs to be in "
                "(0.0, 1.0].", threshold)
            return False

        return True

    def estimate(self, pts1, pts2, inlier_scale=1.0):
        import cv2

        if len(pts1) < MIN_FUNDAMENTAL_POINTS or \
                len(pts2) < MIN_FUNDAMENTAL_POINTS:
            logger.error("Not enough points to estimate a fundamental matrix")
            return None, []

        matrix, mask = cv2.findFundamentalMat(
            _points(pts1), _points(pts2), cv2.FM_RANSAC, inlier_scale,
            self._confidence_threshold)

        if matrix is None:
            return None, _flags(mask, len(pts1))

        # `findFundamentalMat` can return three stacked solutions from the
        # seven point algorithm. The C++ handed whatever came back to a 3x3
        # conversion that threw on a shape mismatch, so the first is what a
        # caller ever saw work; taking it explicitly says so.
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            matrix = matrix[:3, :3]

        return FundamentalMatrixD(matrix), _flags(mask, len(pts1))


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        EstimateHomographyOCV, "ocv",
        "Use OpenCV to estimate a homography from feature matches.")
    register_vital_algorithm(
        EstimateFundamentalMatrixOCV, "ocv",
        "Use OpenCV to estimate a fundamental matrix from feature matches.")
