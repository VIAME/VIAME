# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Descriptor matching with FLANN, on cv2.

`library/image_processing/match_features_flannbased.cxx` in python. The index
is OpenCV's and so is the cross-check rule; only the language changes.

Worth knowing before relying on it: **this matcher is not deterministic**.
`cv::FlannBasedMatcher` builds randomised KD-trees, and OpenCV seeds them
from the clock, so the same descriptors matched twice in one process give
different answers -- 45 or 46 pairs out of 81 descriptors, in the recording
this was ported against. `tests/golden/opencv` compares it on agreement
rather than on bytes for that reason, and everything downstream of it
inherits the same looseness.
"""

import logging

import numpy as np

from kwiver.vital.algo import MatchFeatures
from kwiver.vital.types import MatchSet

from viame.image_processing.ocv_feature_types import (matches_to_set,
                                                      set_to_descriptors)

logger = logging.getLogger(__name__)

# `cv::flann::LshIndexParams( 12, 20, 2 )`, which is what the C++ passed for a
# binary descriptor. Ordinary float descriptors get the FlannBasedMatcher's
# own default, a randomised KD-tree forest.
LSH_PARAMS = (12, 20, 2)


def _as_bool(value):
    return str(value).strip().lower() in ("true", "yes", "on", "1")


class MatchFeaturesFlannBased(MatchFeatures):
    """Match descriptors with `cv2.FlannBasedMatcher`."""

    def __init__(self):
        MatchFeatures.__init__(self)

        self._binary_descriptors = False
        self._cross_check = True
        self._cross_check_k = 1
        self._matcher = None

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

        # The C++ rebuilt its matcher on every configuration change: there is
        # no way to change an index's parameters in place.
        self._matcher = None

    def check_configuration(self, cfg):
        k = int(float(cfg.get_value("cross_check_k", str(self._cross_check_k))
                      or self._cross_check_k))

        if k == 0:
            logger.error("Cross-check K value must be greater than 0.")
            return False

        return True

    # ------------------------------------------------------------------

    def _flann(self):
        import cv2

        if self._matcher is None:
            if self._binary_descriptors:
                self._matcher = cv2.FlannBasedMatcher(
                    dict(algorithm=6, table_number=LSH_PARAMS[0],
                         key_size=LSH_PARAMS[1],
                         multi_probe_level=LSH_PARAMS[2]))
            else:
                self._matcher = cv2.FlannBasedMatcher()

        return self._matcher

    def _cross_check_match(self, first, second):
        """Keep a match only if it is mutual within the top k.

        The C++ `cross_check_match`, and its order of results with it: one
        match per query descriptor at most, taken in query order, and the
        first forward candidate that has any backward candidate pointing home
        wins.
        """
        matcher = self._flann()

        forward = matcher.knnMatch(first, second, self._cross_check_k)
        backward = matcher.knnMatch(second, first, self._cross_check_k)

        kept = []
        for candidates in forward:
            for candidate in candidates:
                mutual = any(back.trainIdx == candidate.queryIdx
                             for back in backward[candidate.trainIdx])
                if mutual:
                    kept.append(candidate)
                    break

        return kept

    def match(self, feat1, desc1, feat2, desc2):
        if desc1 is None or desc2 is None:
            return None

        if desc1.size() == 0 or desc2.size() == 0:
            return None

        first = set_to_descriptors(desc1, self._binary_descriptors)
        second = set_to_descriptors(desc2, self._binary_descriptors)

        if first is None or second is None:
            logger.debug("Unable to convert descriptors to OpenCV format")
            return None

        if self._cross_check:
            matches = self._cross_check_match(first, second)
        else:
            matches = self._flann().match(first, second)

        return matches_to_set(matches)


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        MatchFeaturesFlannBased, "ocv_flann_based",
        "OpenCV feature matcher using FLANN (Approximate Nearest Neighbors)")
