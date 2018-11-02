###
# Loadable algorithms

find_package( Caffe REQUIRED )

if( NOT VIAME_ENABLE_CUDA )
  add_definitions( -DCPU_ONLY )
endif()

include_directories( "${Caffe_INCLUDE_DIRS}" )
link_directories( "${Caffe_LIBRARY_DIR}" )

set( plugin_headers
  faster_rcnn_detector.h
  )

set( plugin_sources
  faster_rcnn_detector.cxx
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
  PUBLIC               vital kwiversys
                       vital_exceptions
                       kwiver_algo_ocv
                       ${Caffe_LIBRARIES}
  )

set_target_properties( viame_faster_rcnn PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

algorithms_create_plugin( viame_faster_rcnn
  register_algorithms.cxx
  )
