
# VIAME ONNX plugins, installed as the "viame.onnx" kwiver python package.
#
# The generic onnxruntime detector is `library/object_detectors/onnx` and the
# classifier and reclassifier are `library/classifiers/onnx` since P2-T05.
# What is left is the stereo tooling, which P2-T06 takes to `measurement`:
#   * ONNX stereo utilities: epipolar template-matching / foundation-stereo
#     correspondence exported to self-contained .onnx graphs (torch for export
#     only; numpy, opencv, and optionally scipy for .mat calibration).

viame_create_python_init( onnx )

set( onnx_modules
  __init__
  epipolar_matcher
  epipolar_dino_matcher
  triangulate
  calibration_io
  geometry_numpy
  export_stereo_mapping
  run_epipolar_onnx
  fast_foundation_stereo )

foreach( module ${onnx_modules} )
  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/${module}.py
    onnx
    ${module} )
endforeach()
