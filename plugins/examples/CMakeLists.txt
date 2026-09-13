###
# Loadable algorithms

set( plugin_headers
  hello_world_detector.h
  hello_world_filter.h
  )

set( plugin_sources
  hello_world_detector.cxx
  hello_world_filter.cxx
  )

viame_install_headers(
  SUBDIR     viame
  ${plugin_headers}
  )

viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_examples_export.h
  NOPATH   SUBDIR     viame
  )

viame_add_library( viame_examples
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_examples PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

target_link_libraries( viame_examples
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
                       ${OpenCV_LIBRARIES}
  )

set_target_properties( viame_examples PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

viame_register_statically( viame_examples
  register_algorithms.cxx
  )

if( VIAME_ENABLE_PYTHON )
  viame_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    examples
    __init__)

  viame_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/hello_world_detector.py
    examples
    hello_world_detector)

  viame_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/hello_world_filter.py
    examples
    hello_world_filter)
endif()
