###
# Loadable algorithms

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/maskcut_detector.py
  arrows/pytorch
  maskcut_detector)
