
kwiver_create_python_init( vertex-ai )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  vertex-ai
  __init__ )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/app.py
  vertex-ai
  app )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/train_handler.py
  vertex-ai
  train_handler )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/process_handler.py
  vertex-ai
  process_handler )

install(
  FILES Dockerfile requirements.txt
  DESTINATION share/viame/vertex-ai )
