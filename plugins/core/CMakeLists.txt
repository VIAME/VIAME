# Build / Install Plugin containing core algorithm implementations

set( plugin_headers
  detected_object_set_input_habcam.h
  )

set( plugin_sources
  detected_object_set_input_habcam.cxx

  register_algorithms.h    register_algorithms.cxx
  )

kwiver_install_headers(
  SUBDIR     viame
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_core_export.h
  NOPATH   SUBDIR     viame
  )

kwiver_add_library( viame_core
  ${plugin_headers}
  ${plugin_sources}
  )

target_link_libraries( viame_core
  PUBLIC               vital
  )

algorithms_create_plugin( viame_core
  algorithm_plugin_interface.cxx
  )

if (KWIVER_ENABLE_TESTS)
#  add_subdirectory(tests)
endif()
