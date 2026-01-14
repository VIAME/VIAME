
kwiver_create_python_init( tensorflow )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/tf_detector.py
  tensorflow
  tf_detector )
