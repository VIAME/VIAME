# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""The generic onnxruntime object detector -- no torch at inference.

Runs any detection graph described by a `.modelspec.json` sidecar. From
`viame.onnx` in P2-T05; a subpackage of its own so that a build without
`VIAME_ENABLE_ONNX` neither installs nor declares it.
"""

# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
__vital_algorithm_declarations__ = [
    ( "image_object_detector", "onnx",
      "Generic ONNX object detector (onnxruntime, no torch)",
      "viame.object_detectors.onnx.onnx_detector:OnnxDetector" ),
]
