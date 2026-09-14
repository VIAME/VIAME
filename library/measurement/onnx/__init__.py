# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""The ONNX stereo tools.

The epipolar template-matching and Fast-Foundation-Stereo correspondence,
exported to self-contained .onnx graphs, and their runner: torch for export
only; numpy, opencv and optionally scipy to run. From `viame.onnx` in P2-T08,
installed only with `VIAME_ENABLE_ONNX`. The generic onnxruntime detector is
`viame.object_detectors.onnx` and the classifiers `viame.classifiers.onnx`.
"""

# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
__vital_algorithm_declarations__ = [
    ( "compute_stereo_depth_map", "fast_foundation_stereo_onnx",
      "Stereo depth/disparity estimation using NVIDIA Fast-Foundation-Stereo ONNX/TensorRT export",
      "viame.measurement.onnx.fast_foundation_stereo:FastFoundationStereoOnnx" ),
]

__sprokit_process_declarations__ = []
