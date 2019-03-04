
###
# Loadable algorithms

find_package( OpenCV REQUIRED )

include_directories( "${OpenCV_INCLUDE_DIRS}" )
link_directories( "${OpenCV_LIBRARY_DIR}" )

set( plugin_headers
  ocv_image_enhancement.h
  ocv_debayer_filter.h
  ocv_stereo_depth_map.h
  )

set( plugin_sources
  ocv_image_enhancement.cxx
  ocv_debayer_filter.cxx
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

get_target_property( results viame_opencv COMPILE_DEFINITIONS )

if( VIAME_OPENCV_VERSION MATCHES "2.*" )
  target_compile_definitions( viame_opencv PRIVATE -DVIAME_OPENCV_VER_2 )
endif()

get_target_property( results viame_opencv COMPILE_DEFINITIONS )

target_link_libraries( viame_opencv
  PUBLIC               vital
                       vital_algo
                       vital_config
                       vital_exceptions
                       vital_logger
                       vital_util
                       kwiversys
                       kwiver_algo_ocv
                       ${OpenCV_LIBRARIES}
  )

set_target_properties( viame_opencv PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

algorithms_create_plugin( viame_opencv
  register_algorithms.cxx
  )

target_link_libraries( viame_opencv_plugin
  PUBLIC               vital_vpm
  )

if( VIAME_ENABLE_PYTHON )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    viame/processes/opencv
    __init__ )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/ocv_multimodal_registration.py
    viame/processes/opencv
    ocv_multimodal_registration )

endif()
