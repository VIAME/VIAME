
kwiver_create_python_init( viame/arrows/tensorflow )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/tf_detector.py
  viame/arrows/tensorflow
  tf_detector )
