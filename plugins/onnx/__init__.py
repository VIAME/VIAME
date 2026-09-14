# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""VIAME ONNX plugins (installed as the ``viame.onnx`` package).

The epipolar / foundation stereo ONNX utilities, declared below rather than
imported. The generic detector is `viame.object_detectors.onnx` and the
classifiers are `viame.classifiers.onnx` since P2-T05.
"""

# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
# `kwiver.vital.plugins.discovery` turns each into a stand-in that imports
# its module the first time something asks for an instance.
#
# This replaced a `__vital_algorithm_register__` that imported every
# implementation module at startup, which was the only way the classes came
# to exist for the subclass walk that registers them. See P8-T10.
__vital_algorithm_declarations__ = [
    ( "compute_stereo_depth_map", "fast_foundation_stereo_onnx",
      "Stereo depth/disparity estimation using NVIDIA Fast-Foundation-Stereo ONNX/TensorRT export",
      "viame.onnx.fast_foundation_stereo:FastFoundationStereoOnnx" ),
]
