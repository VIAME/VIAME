###
# @template_dir@
#
# A library of its own under `library/`, named for what it does. Add
# `add_subdirectory( @template_dir@ )` to `library/CMakeLists.txt`.
##

set( plugin_headers
  @template@_detector.h
  )

set( plugin_sources
  @template@_detector.cxx
  )

viame_install_headers(
  SUBDIR     viame/@template_dir@
  ${plugin_headers}
  )

viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_@template_lib@_export.h
  NOPATH   SUBDIR     viame/@template_dir@
  )

viame_add_library( viame_@template_lib@
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_@template_lib@ PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

target_link_libraries( viame_@template_lib@
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
  )

set_target_properties( viame_@template_lib@ PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

viame_register_statically( viame_@template_lib@
  register_algorithms.cxx
  )
