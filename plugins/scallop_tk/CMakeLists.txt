###
# Loadable algorithms

find_package( ScallopTK REQUIRED )

## These should come from the find package.
set( ScallopTK_LIBS ScallopTK )

include_directories( "${ScallopTK_INCLUDE_DIR}" )
link_directories( "${ScallopTK_LIBRARY_DIR}" )

set( plugin_headers
  scallop_detector.h
  )

set( plugin_sources
  scallop_detector.cxx

  register_algorithms.h    register_algorithms.cxx
  )

kwiver_install_headers(
  SUBDIR     viame
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_algo_export.h
  NOPATH   SUBDIR     viame
  )

kwiver_add_library( viame_algo
  ${plugin_headers}
  ${plugin_sources}
  )

target_link_libraries( viame_algo
  PUBLIC               vital
                       kwiver_algo_ocv
                       ScallopTK
  )

algorithms_create_plugin( viame_algo
  algorithm_plugin_interface.cxx
  )
