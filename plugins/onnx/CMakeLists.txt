
# ONNX stereo plugins: epipolar template-matching stereo correspondence
# exported to a single self-contained .onnx graph.
#
# These are pure-Python utilities (export + run + calibration/geometry helpers)
# installed as the "onnx_stereo" kwiver python package. They depend on torch
# (export only), onnxruntime (run), numpy, opencv, and optionally scipy (.mat
# calibration).

kwiver_create_python_init( onnx_stereo )

set( onnx_stereo_modules
  __init__
  epipolar_matcher
  triangulate
  calibration_io
  geometry_numpy
  export_onnx
  run_epipolar_onnx )

foreach( module ${onnx_stereo_modules} )
  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/${module}.py
    onnx_stereo
    ${module} )
endforeach()
