###
# Loadable algorithms

set( plugin_headers
  @template@_detector.h
  )

set( plugin_sources
  @template@_detector.cxx

  register_algorithms.h    register_algorithms.cxx
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
  PUBLIC               vital
  )

algorithms_create_plugin( viame_@template_lib@
  algorithm_plugin_interface.cxx
  )
