###
# Loadable algorithms

find_package( Caffe REQUIRED )

include_directories( "${Caffe_INCLUDE_DIRS}" )
link_directories( "${Caffe_LIBRARY_DIR}" )

set( plugin_headers
  faster_rcnn_detector.h
  )

set( plugin_sources
  faster_rcnn_detector.cxx

  register_algorithms.h    register_algorithms.cxx
  )

kwiver_install_headers(
  SUBDIR     viame
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_faster_rcnn_export.h
  NOPATH   SUBDIR     viame
  )

kwiver_add_library( viame_faster_rcnn
  ${plugin_headers}
  ${plugin_sources}
  )

target_link_libraries( viame_faster_rcnn
  PUBLIC               vital
                       kwiver_algo_ocv
  )

algorithms_create_plugin( viame_faster_rcnn
  algorithm_plugin_interface.cxx
  )
