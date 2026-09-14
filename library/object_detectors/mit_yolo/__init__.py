# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""The MIT YOLO detector.

From `viame.pytorch` in P2-T05. Installed only with `VIAME_ENABLE_PYTORCH-MIT-YOLO`, so its
declarations exist only in a build that has it.
"""

# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
__vital_algorithm_declarations__ = [
    ( "image_object_detector", "mit_yolo",
      "PyTorch MIT YOLO detection routine",
      "viame.object_detectors.mit_yolo.mit_yolo_detector:MITYoloDetector" ),
]
