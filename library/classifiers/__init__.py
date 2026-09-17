# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Refiners, classifiers and detection mergers, in python.

From `viame.core` in P2-T05. `detection_fusion_core` is the shared
machinery the mergers below are thin wrappers around, and is not itself
registered.
"""


# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
# `viame.plugins.discovery` turns each into a stand-in that imports
# its module the first time something asks for an instance.
__vital_algorithm_declarations__ = [
    ( "merge_detections", "coverage_reinforce",
      "Reinforce detections using a weakly-localizing evidence source",
      "viame.classifiers.merge_detections_coverage_reinforce:MergeDetectionsCoverageReinforce" ),
    ( "merge_detections", "merge",
      "Concatenate all input detection sets without resolving overlaps",
      "viame.classifiers.merge_detections_simple:MergeDetectionsMerge" ),
    ( "merge_detections", "nms_fusion",
      "Fusion of multiple different detections",
      "viame.classifiers.merge_detections_nms_fusion:MergeDetectionsNMSFusion" ),
    ( "merge_detections", "simple",
      "Concatenate all input detection sets without resolving overlaps",
      "viame.classifiers.merge_detections_simple:MergeDetectionsSimple" ),
]


# ----------------------------------------------------------------------------
# The processes this package provides, without importing any of them.
#
# Each entry is ( name, description, "module:Class" ).
__sprokit_process_declarations__ = [
    ( "multicam_homog_det_suppressor",
      "Multi-camera homography-based detection suppressor",
      "viame.classifiers.multicam_homog_det_suppressor:MulticamHomogDetSuppressor" ),
]
