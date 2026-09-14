# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""The onnxruntime classifiers -- no torch at inference.

The whole-frame classifier and the detection reclassifier, the onnxruntime
replacements for `netharn_classifier` and the netharn refiner. From
`viame.onnx` in P2-T05; a subpackage of its own so that a build without
`VIAME_ENABLE_ONNX` neither installs nor declares them.
"""

# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
__vital_algorithm_declarations__ = [
    ( "image_object_detector", "onnx_classifier",
      "Whole-frame ONNX classifier (onnxruntime, no torch)",
      "viame.classifiers.onnx.onnx_classifier:OnnxClassifier" ),
    ( "refine_detections", "onnx",
      "ONNX detection reclassifier (onnxruntime, no torch)",
      "viame.classifiers.onnx.onnx_refiner:OnnxRefiner" ),
]
