###
# Loadable algorithms

find_package( ScallopTK REQUIRED )

## These should come from the find package.
set( ScallopTK_LIBS ScallopTK )

include_directories( "${ScallopTK_INCLUDE_DIR}" )
link_directories( "${ScallopTK_LIBRARY_DIR}" )

set( plugin_headers
  scallop_tk_detector.h
  )

set( plugin_sources
  scallop_tk_detector.cxx
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

target_include_directories( viame_scallop_tk PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

target_link_libraries( viame_scallop_tk
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::kwiver_algo_ocv
                       opencv_core opencv_legacy
                       ScallopTK
  )

set_target_properties( viame_scallop_tk PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

if( VIAME_ENABLE_CAFFE AND WIN32 )
  target_link_libraries( viame_scallop_tk
    PUBLIC             caffe glog )
endif()

if( VIAME_ENABLE_CUDA )
  target_link_libraries( viame_scallop_tk
    PUBLIC             ${CUDA_CUDA_LIBRARY}
                       ${CUDA_LIBRARIES}
                       ${CUDA_CUBLAS_LIBRARIES}
                       ${CUDA_curand_LIBRARY} )
endif()

algorithms_create_plugin( viame_scallop_tk
  register_algorithms.cxx
  )

target_link_libraries( viame_scallop_tk_plugin
  PUBLIC               kwiver::vital_vpm
  )
