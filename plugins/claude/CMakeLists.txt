###
# Utilities for running VIAME tools under claude (LLM) supervision
##

set( plugin_headers
  train_supervisor.h
  )

set( plugin_sources
  train_supervisor.cxx
  )

kwiver_install_headers(
  SUBDIR     viame
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_claude_export.h
  NOPATH   SUBDIR     viame
  )

kwiver_add_library( viame_claude
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_claude PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

target_link_libraries( viame_claude
  PUBLIC               kwiver::vital
                       kwiver::vital_logger
                       kwiver::vital_util
                       kwiver::kwiversys
                       viame_core
  )

find_package( Threads REQUIRED )

target_link_libraries( viame_claude
  PRIVATE              Threads::Threads
  )

set_target_properties( viame_claude PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )
