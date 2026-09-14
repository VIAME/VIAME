# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Learned stereo depth: NVIDIA Foundation-Stereo and its real-time variant.

From `viame.pytorch` in P2-T06. Installed only with
`VIAME_ENABLE_PYTORCH-STEREO`, so these declarations exist only in a build
that has it. The ONNX export of the real-time variant is P2-T08's.
"""


# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
__vital_algorithm_declarations__ = [
    ( "compute_stereo_depth_map", "fast_foundation_stereo",
      "Stereo depth/disparity estimation using NVIDIA Fast-Foundation-Stereo (real-time variant)",
      "viame.measurement.stereo.fast_foundation_stereo:FastFoundationStereo" ),
    ( "compute_stereo_depth_map", "foundation_stereo",
      "Stereo depth/disparity estimation using NVIDIA Foundation-Stereo model",
      "viame.measurement.stereo.foundation_stereo:FoundationStereo" ),
]
