
###
# Python Flask application (Vertex AI custom container)
##

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

###
# C++ sprokit processes (Vertex AI REST client)
##

find_package( OpenSSL QUIET )

if( OpenSSL_FOUND )

  set( plugin_headers
    vertex_ai_client.h
    vertex_ai_detector.h
    vertex_ai_trainer.h
  )

  set( plugin_sources
    vertex_ai_client.cxx
    vertex_ai_detector.cxx
    vertex_ai_trainer.cxx
  )

  kwiver_add_library( viame_vertex_ai
    ${plugin_headers}
    ${plugin_sources}
  )

  target_compile_definitions( viame_vertex_ai PRIVATE
    CPPHTTPLIB_OPENSSL_SUPPORT
  )

  target_link_libraries( viame_vertex_ai PUBLIC
    kwiver::vital
    kwiver::vital_config
    kwiver::vital_logger
    OpenSSL::SSL
    OpenSSL::Crypto
  )

  target_include_directories( viame_vertex_ai PRIVATE
    ${VIAME_HTTPLIB_INCLUDE_DIR}
  )

  kwiver_add_plugin( viame_processes_vertex_ai
    SUBDIR ${viame_plugin_process_subdir}
    SOURCES register_processes.cxx
    PRIVATE viame_vertex_ai
            kwiver::sprokit_pipeline
            kwiver::vital
            kwiver::vital_util
  )

  target_include_directories( viame_processes_vertex_ai PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

endif()
