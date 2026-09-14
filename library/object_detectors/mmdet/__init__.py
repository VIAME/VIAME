# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""MMDetection inference, and the config compatibility shim it runs through.

From `viame.pytorch` in P2-T05. Installed only with `VIAME_ENABLE_PYTORCH-MMDET`, so its
declarations exist only in a build that has it.
"""

# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
__vital_algorithm_declarations__ = [
    ( "image_object_detector", "mmdet",
      "PyTorch MMDetection inference routine",
      "viame.object_detectors.mmdet.mmdet_detector:MMDetDetector" ),
    ( "train_detector", "mmdet",
      "PyTorch MMDetection training routine",
      "viame.object_detectors.mmdet.mmdet_trainer:MMDetTrainer" ),
]
