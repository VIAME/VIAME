
###
# Loadable algorithms
##

# No OpenCV, and not much else either: P2-T05 took the colour and split
# filters, `add_keypoints_from_mask`, `detect_in_subregions` and the windowed
# detector and refiner into `library/`. What is left is the stereo pairing
# and measurement, which P2-T06 takes, and the windowed trainer, which waits
# for `library/training`.

set( plugin_headers
  pair_stereo_detections.h
  pair_stereo_tracks.h
  windowed_trainer.h
  windowed_utils.h
)

set( plugin_sources
  pair_stereo_detections.cxx
  pair_stereo_tracks.cxx
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


# Add process library
set( process_headers
  measure_objects_process.h
  pair_stereo_detections_process.h
  pair_stereo_tracks_process.h
)

set( process_sources
  register_processes.cxx
  measure_objects_process.cxx
  pair_stereo_detections_process.cxx
  pair_stereo_tracks_process.cxx
)

viame_add_plugin( viame_processes_opencv
  KIND        PROCESSES
  SOURCES          ${process_sources}
  PRIVATE          kwiver::vital
                   kwiver::vital_algo
                   kwiver::vital_config
                   kwiver::vital_logger
                   kwiver::vital_util
                   kwiver::vital_vpm
                   kwiver::sprokit_pipeline
                   viame_measurement
                   viame_opencv
                   # `measure_objects_process` hangs keypoints off a mask
                   viame_segmentation
)

target_include_directories( viame_processes_opencv PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/plugins>
  )

if( VIAME_ENABLE_PYTHON )
  set( _rel_modnames
    __init__
    prior_coverage_opencv
    stereo_demo
    stereo_expt
    stereo_pipeline
    stereo_utils
  )

  foreach( rel_modname IN ITEMS ${_rel_modnames} )
    string( REPLACE "." "/" rel_modpath_noext ${rel_modname} )
    viame_add_python_module(
      "${CMAKE_CURRENT_SOURCE_DIR}/${rel_modpath_noext}.py"
      "opencv"
      "${rel_modname}" )
  endforeach()
endif()

