
###
# Loadable algorithms
##

# No OpenCV, and almost nothing else: P2-T05 took the filters, detectors and
# segmenter, and P2-T06 the stereo pairing and measurement, into `library/`.
# What is left is the windowed trainer, which waits for `library/training`,
# and the stereo demo scripts lite-library-layout.md deletes with the
# directory in P2-T08.

set( plugin_headers
  windowed_trainer.h
  windowed_utils.h
)

set( plugin_sources
  windowed_trainer.cxx
  windowed_utils.cxx
)

viame_install_headers(
  SUBDIR     viame
  ${plugin_headers}
  )

viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_opencv_export.h
  NOPATH   SUBDIR     viame
  )

viame_add_library( viame_opencv
  ${plugin_headers}
  ${plugin_sources}
  )

target_link_libraries( viame_opencv
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
                       viame_core
                       viame_measurement
                       # the in-house kernels the ported filters run on
                       viame_image_ops
                       # the calibration writers, since P7-T05 replaced
                       # cv::FileStorage with library/file_io/opencv_yaml
                       viame_file_io
  )

target_include_directories( viame_opencv PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

set_target_properties( viame_opencv PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

viame_register_statically( viame_opencv
  register_algorithms.cxx
  )

if( VIAME_ENABLE_PYTHON )
  set( _rel_modnames
    __init__
    stereo_demo
    stereo_expt
  )

  foreach( rel_modname IN ITEMS ${_rel_modnames} )
    string( REPLACE "." "/" rel_modpath_noext ${rel_modname} )
    viame_add_python_module(
      "${CMAKE_CURRENT_SOURCE_DIR}/${rel_modpath_noext}.py"
      "opencv"
      "${rel_modname}" )
  endforeach()
endif()

