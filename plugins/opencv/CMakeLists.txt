
###
# Loadable algorithms

find_package( OpenCV REQUIRED )

include_directories( "${OpenCV_INCLUDE_DIRS}" )
link_directories( "${OpenCV_LIBRARY_DIR}" )

set( plugin_headers
  ocv_stereo_depth_map.h
  )

set( plugin_sources
  ocv_stereo_depth_map.cxx
  )

kwiver_install_headers(
  SUBDIR     viame
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_opencv_export.h
  NOPATH   SUBDIR     viame
  )

kwiver_add_library( viame_opencv
  ${plugin_headers}
  ${plugin_sources}
  )

target_link_libraries( viame_opencv
  PUBLIC               vital kwiversys
                       kwiver_algo_ocv
                       ${OpenCV_LIBRARIES}
  )

algorithms_create_plugin( viame_opencv
  register_algorithms.cxx
  )
