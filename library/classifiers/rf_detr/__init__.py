# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""The RF-DETR refiner, which runs the detector's segmentation and keypoint
heads on boxes it is given.

From `viame.pytorch` in P2-T05. Installed only with `VIAME_ENABLE_PYTORCH-RF-DETR`, so its
declarations exist only in a build that has it.
"""

# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
__vital_algorithm_declarations__ = [
    ( "refine_detections", "rf_detr",
      "Run RF-DETR segmentation/keypoint heads on existing boxes",
      "viame.classifiers.rf_detr.rf_detr_refiner:RFDETRRefiner" ),
]
