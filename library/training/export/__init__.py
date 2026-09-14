# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""ONNX export: the `convert_to_onnx` process.

From `plugins/pytorch` in P2-T07. Installed only with `VIAME_ENABLE_ONNX`. The
exporters it dispatches to are beside their backends -- `yolomit_to_onnx` in
`object_detectors/mit_yolo`, `netharn_mmdet_to_onnx` in
`object_detectors/netharn`, `netharn_clf_to_onnx` in `classifiers/netharn`,
`rf_detr_to_onnx` in `object_detectors/rf_detr` -- except `darknet_to_onnx`,
which is here, gated on `VIAME_ENABLE_DARKNET`, until darknet moves in P2-T08.

`onnx_exporters/__init__.py` imported all four exporters; it was never
installed and nothing used it, so it is not reproduced.
"""

# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
__vital_algorithm_declarations__ = []

__sprokit_process_declarations__ = [
    ( "convert_to_onnx",
      "Convert a VIAME model to onnx",
      "viame.training.export.convert_to_onnx_process:OnnxConverter" ),
]
