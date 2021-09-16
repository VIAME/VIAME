###
# Example CMakeLists file for generating plugins
##

set( plugin_headers
  @template@_detector.h
  )

set( plugin_sources
  @template@_detector.cxx
  )

kwiver_install_headers(
  SUBDIR     viame
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_@template_lib@_export.h
  NOPATH   SUBDIR     viame
  )

kwiver_add_library( viame_@template_lib@
  ${plugin_headers}
  ${plugin_sources}
  )

target_link_libraries( viame_@template_lib@
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
  )

algorithms_create_plugin( viame_@template_lib@
  register_algorithms.cxx
  )
