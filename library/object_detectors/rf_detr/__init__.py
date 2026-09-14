# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""The RF-DETR detector. Its refiner is `viame.classifiers.rf_detr`.

From `viame.pytorch` in P2-T05. Installed only with `VIAME_ENABLE_PYTORCH-RF-DETR`, so its
declarations exist only in a build that has it.
"""

# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
__vital_algorithm_declarations__ = [
    ( "image_object_detector", "rf_detr",
      "PyTorch RF-DETR detection routine",
      "viame.object_detectors.rf_detr.rf_detr_detector:RFDETRDetector" ),
    ( "train_detector", "rf_detr",
      "PyTorch RF-DETR detection training routine",
      "viame.object_detectors.rf_detr.rf_detr_trainer:RFDETRTrainer" ),
]
