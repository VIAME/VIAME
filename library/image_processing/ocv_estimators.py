# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""RANSAC estimation of a homography and a fundamental matrix.

`library/image_processing/estimate_{homography,fundamental_matrix}.cxx` in
python. These were the last two call sites of `cv::findHomography` and
`cv::findFundamentalMat`; both are `viame.utilities.geometry`'s now -- a
normalised DLT and Hartley's eight point algorithm, each inside a RANSAC
loop with a refit over the consensus set.

Neither reproduces OpenCV's numbers, and neither is meant to: a RANSAC
draws different samples and converges on a different member of the set of
models the data supports. What they are held to is accuracy against known
answers, and `tests/golden/opencv`'s recordings for these two cases were
re-made against them deliberately. On synthetic correspondences with a
quarter corrupted, measured before the switch:

    homography   0.1092 px of reprojection error, against OpenCV's 0.1090,
                 and the identical inlier set
    fundamental  0.28 px of Sampson error against OpenCV's 0.72, and 99%
                 agreement with the true inlier set against its 91%

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

from viame.algo import EstimateFundamentalMatrix, EstimateHomography
from viame.types import FundamentalMatrixD, HomographyD
from viame.utilities import geometry

logger = logging.getLogger(__name__)

# The C++ refused below these and said so, rather than letting OpenCV assert.
MIN_HOMOGRAPHY_POINTS = 4
MIN_FUNDAMENTAL_POINTS = 8


def _points(values):
    """A vital point list as an N by 2 array.

    Float32, still: the C++ built `std::vector<cv::Point2f>`, so the
    estimate is made from single-precision coordinates and would move
    slightly if this passed doubles. The estimators promote to float64
    internally, which is where the arithmetic wants to be, but what they are
    given is what the C++ was given.
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
    """Estimate a homography with `geometry.find_homography` and RANSAC."""

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
        if len(pts1) < MIN_HOMOGRAPHY_POINTS or \
                len(pts2) < MIN_HOMOGRAPHY_POINTS:
            logger.error("Not enough points to estimate a homography")
            return None, []

        matrix, mask = geometry.find_homography(
            _points(pts1), _points(pts2), threshold=inlier_scale)

        if matrix is None:
            return None, []

        return HomographyD(np.asarray(matrix, dtype=np.float64)), \
            _flags(mask, len(pts1))


class EstimateFundamentalMatrixOCV(EstimateFundamentalMatrix):
    """Estimate a fundamental matrix with `geometry.find_fundamental`."""

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
        if len(pts1) < MIN_FUNDAMENTAL_POINTS or \
                len(pts2) < MIN_FUNDAMENTAL_POINTS:
            logger.error("Not enough points to estimate a fundamental matrix")
            return None, []

        matrix, mask = geometry.find_fundamental(
            _points(pts1), _points(pts2), threshold=inlier_scale,
            confidence=self._confidence_threshold)

        if matrix is None:
            return None, _flags(mask, len(pts1))

        # The stacked-solutions case is gone with the seven point algorithm:
        # `find_fundamental` is eight point and returns one three by three.
        return FundamentalMatrixD(np.asarray(matrix, dtype=np.float64)), \
            _flags(mask, len(pts1))


def __vital_algorithm_register__():
    from viame.utilities.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        EstimateHomographyOCV, "ocv",
        "Use OpenCV to estimate a homography from feature matches.")
    register_vital_algorithm(
        EstimateFundamentalMatrixOCV, "ocv",
        "Use OpenCV to estimate a fundamental matrix from feature matches.")
