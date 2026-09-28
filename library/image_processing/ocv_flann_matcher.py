# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Descriptor matching, `ocv_flann_based` without FLANN.

`library/image_processing/match_features_flannbased.cxx` in python, and off
cv2. The cross-check rule and the order of its results are still OpenCV's; the
index is not, and **that is an improvement rather than a compromise**.

`cv::FlannBasedMatcher` builds randomised KD-trees and searches a bounded
number of leaves, so it is both *approximate* and *not deterministic*: OpenCV
seeds the trees from the clock, and the same descriptors matched twice in one
process gave different answers -- 45 or 46 pairs out of 81 descriptors, in the
recording this was first ported against. Everything downstream inherited that,
which is why `tests/reference/opencv` compares this matcher on agreement rather
than on bytes.

This searches every candidate instead. Exhaustive nearest neighbours over a few
thousand descriptors is a matrix multiply, and at the sizes a frame pair
produces it is not the bottleneck -- the shipped stabilizer pipelines match a
few hundred to a few thousand per frame. So the answer is the *true* nearest
neighbour, the same one every run, and the name is kept because the configs ask
for it: `ocv_flann_based` now means "the matcher those configs select", not
"FLANN".

The cost is that a caller wanting an approximate index for a very large set no
longer has one here. Nothing in the tree wants that: colmap does its own
matching, and the registration utilities match per frame pair.
"""

import logging

import numpy as np

from viame.algo import MatchFeatures
from viame.image_processing.matching import as_matrix, match
from viame.types import MatchSet

logger = logging.getLogger(__name__)


def _as_bool(value):
    return str(value).strip().lower() in ("true", "yes", "on", "1")


class MatchFeaturesFlannBased(MatchFeatures):
    """Match descriptors by exhaustive nearest neighbour."""

    def __init__(self):
        MatchFeatures.__init__(self)

        self._binary_descriptors = False
        self._cross_check = True
        self._cross_check_k = 1

    # ------------------------------------------------------------------
    # Configuration

    def get_configuration(self):
        cfg = super(MatchFeatures, self).get_configuration()
        cfg.set_value("binary_descriptors",
                      "true" if self._binary_descriptors else "false")
        cfg.set_value("cross_check", "true" if self._cross_check else "false")
        cfg.set_value("cross_check_k", str(int(self._cross_check_k)))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        self._binary_descriptors = _as_bool(cfg.get_value("binary_descriptors"))
        self._cross_check = _as_bool(cfg.get_value("cross_check"))
        self._cross_check_k = int(float(cfg.get_value("cross_check_k")))

    def check_configuration(self, cfg):
        k = int(float(cfg.get_value("cross_check_k", str(self._cross_check_k))
                      or self._cross_check_k))

        if k == 0:
            logger.error("Cross-check K value must be greater than 0.")
            return False

        return True

    # ------------------------------------------------------------------

    def _cross_check_match(self, first, second):
        """Keep a match only if it is mutual within the top k.

        The C++ `cross_check_match`, and its order of results with it: one
        match per query descriptor at most, taken in query order, and the first
        forward candidate with any backward candidate pointing home wins. Note
        it is "any of the backward neighbours", not "the best backward
        neighbour" -- with `cross_check_k` above one that is a much weaker test
        than a strict mutual best.
        """
        return match(first, second, cross_check=True,
                     k=self._cross_check_k, binary=self._binary_descriptors)

    def match(self, feat1, desc1, feat2, desc2):
        if desc1 is None or desc2 is None:
            return None

        if desc1.size() == 0 or desc2.size() == 0:
            return None

        first = as_matrix(desc1, self._binary_descriptors)
        second = as_matrix(desc2, self._binary_descriptors)

        if first is None or second is None:
            logger.debug("Unable to read the descriptors as a matrix")
            return None

        if first.shape[1] != second.shape[1]:
            logger.debug("Descriptor widths differ: %d against %d",
                         first.shape[1], second.shape[1])
            return None

        if self._cross_check:
            matches = self._cross_check_match(first, second)
        else:
            matches = match(first, second, binary=self._binary_descriptors)

        return MatchSet([(int(q), int(t)) for q, t in matches])


def __vital_algorithm_register__():
    from viame.utilities.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        MatchFeaturesFlannBased, "ocv_flann_based",
        "OpenCV feature matcher using FLANN (Approximate Nearest Neighbors)")
