###
# Loadable algorithms

find_package( faster_rcnn REQUIRED )

## These should come from the find package.
set( faster_rcnn_LIBS faster_rcnn )

include_directories( "${faster_rcnn_INCLUDE_DIR}" )
link_directories( "${faster_rcnn_LIBRARY_DIR}" )

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
  ${CMAKE_CURRENT_BINARY_DIR}/viame_scallop_tk_export.h
  NOPATH   SUBDIR     viame
  )

kwiver_add_library( viame_scallop_tk
  ${plugin_headers}
  ${plugin_sources}
  )

target_link_libraries( viame_scallop_tk
  PUBLIC               vital
                       kwiver_algo_ocv
                       faster_rcnn
  )

algorithms_create_plugin( viame_scallop_tk
  algorithm_plugin_interface.cxx
  )
