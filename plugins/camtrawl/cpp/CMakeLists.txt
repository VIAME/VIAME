###
# Loadable algorithms

set( plugin_headers
  camtrawl_detect.h
  )

set( plugin_sources
  camtrawl_detect.cxx
  )

kwiver_install_headers(
  SUBDIR     viame
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_camtrawl_export.h
  NOPATH
  SUBDIR viame
  )

kwiver_add_library( viame_camtrawl
  ${plugin_headers}
  ${plugin_sources}
  )

target_link_libraries( viame_camtrawl
  PUBLIC               vital
                       vital_algo
                       vital_config
                       vital_logger
                       vital_util
                       kwiver_algo_ocv
                       ${OpenCV_LIBRARIES}
                       #opencv_core opencv_legacy
  )

algorithms_create_plugin( viame_camtrawl
  register_algorithms.cxx
  )

target_link_libraries( viame_camtrawl_plugin
  PUBLIC               vital_vpm
  )
