# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""VIAME ONNX plugins (installed as the ``viame.onnx`` package).

Holds the generic onnxruntime object detector plus the epipolar / foundation
stereo ONNX utilities, declared below rather than imported.
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
    ( "image_object_detector", "onnx",
      "Generic ONNX object detector (onnxruntime, no torch)",
      "viame.onnx.onnx_detector:OnnxDetector" ),
    ( "image_object_detector", "onnx_classifier",
      "Whole-frame ONNX classifier (onnxruntime, no torch)",
      "viame.onnx.onnx_classifier:OnnxClassifier" ),
    ( "refine_detections", "onnx",
      "ONNX detection reclassifier (onnxruntime, no torch)",
      "viame.onnx.onnx_refiner:OnnxRefiner" ),
]
