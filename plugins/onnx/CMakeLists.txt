
# VIAME ONNX plugins, installed as the "viame.onnx" kwiver python package.
#
# Two groups of pure-Python, onnxruntime-based tools (no torch at inference):
#   * Generic object detector: onnx_predictor + onnx_detector (kwiver vital
#     algorithm "onnx"), runnable on any detection ONNX graph described by a
#     .modelspec.json sidecar.
#   * Image classifiers: onnx_clf_predictor plus the kwiver algorithms
#     "onnx_classifier" (whole-frame) and the "onnx" refine_detections
#     reclassifier -- the onnxruntime replacements for netharn_classifier and
#     the netharn refiner.
#   * ONNX stereo utilities: epipolar template-matching / foundation-stereo
#     correspondence exported to self-contained .onnx graphs (torch for export
#     only; numpy, opencv, and optionally scipy for .mat calibration).

kwiver_create_python_init( onnx )

set( onnx_modules
  __init__
  onnx_predictor
  onnx_detector
  onnx_clf_predictor
  onnx_classifier
  onnx_refiner
  epipolar_matcher
  epipolar_dino_matcher
  triangulate
  calibration_io
  geometry_numpy
  export_stereo_mapping
  run_epipolar_onnx
  fast_foundation_stereo )

foreach( module ${onnx_modules} )
  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/${module}.py
    onnx
    ${module} )
endforeach()
