# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""SLEAP-NN keypoints on existing detections.

From `viame.pytorch` in P2-T05. Installed only with `VIAME_ENABLE_PYTORCH-SLEAP`, so its
declarations exist only in a build that has it. It was never declared before: `viame.pytorch` declared what the build
that generated its list had enabled, and this option was off there, so
turning it on installed the module and registered nothing.
"""

# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
__vital_algorithm_declarations__ = [
    ( "refine_detections", "sleap",
      "SLEAP-NN keypoints on existing detections",
      "viame.classifiers.sleap.sleap_refiner:SleapRefiner" ),
]
