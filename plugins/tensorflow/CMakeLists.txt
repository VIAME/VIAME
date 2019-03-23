kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  viame/processes/tensorflow
  __init__ )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/tf_detector.py
  viame/processes/tensorflow
  tf_detector )
