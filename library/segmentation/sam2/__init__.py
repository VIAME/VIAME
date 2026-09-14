# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""SAM2 point segmentation, and detection and track refinement.

From `viame.pytorch` in P2-T05. Installed only with `VIAME_ENABLE_PYTORCH-SAM2`, so its
declarations exist only in a build that has it.
"""

# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
__vital_algorithm_declarations__ = [
    ( "refine_detections", "sam2",
      "SAM2-based detection refiner",
      "viame.segmentation.sam2.sam2_refiner:Sam2Refiner" ),
    ( "refine_tracks", "sam2",
      "SAM2-based track refiner for adding segmentation masks to tracks",
      "viame.segmentation.sam2.sam2_refiner:Sam2TrackRefiner" ),
    ( "segment_via_points", "sam2",
      "SAM2-based point segmentation algorithm",
      "viame.segmentation.sam2.sam2_segmenter:SAM2Segmenter" ),
]
