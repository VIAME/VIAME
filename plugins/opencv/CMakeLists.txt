
###
# Loadable algorithms
##

find_package( OpenCV REQUIRED )

include_directories( "${OpenCV_INCLUDE_DIRS}" )
link_directories( "${OpenCV_LIBRARY_DIR}" )

set( plugin_headers
  add_keypoints_from_mask.h
  convert_color_space.h
  debayer_filter.h
  pair_stereo_detections.h
  pair_stereo_tracks.h
  random_hue_shift.h
  split_image_habcam.h
  split_image_horizontally.h
  warp_image_ocv.h
  windowed_detector.h
  windowed_refiner.h
  windowed_trainer.h
  windowed_utils.h
)

set( plugin_sources
  add_keypoints_from_mask.cxx
  convert_color_space.cxx
  debayer_filter.cxx
  pair_stereo_detections.cxx
  pair_stereo_tracks.cxx
  random_hue_shift.cxx
  split_image_habcam.cxx
  split_image_horizontally.cxx
  warp_image_ocv.cxx
  windowed_detector.cxx
  windowed_refiner.cxx
  windowed_trainer.cxx
  windowed_utils.cxx
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

if( "${OpenCV_VERSION}" VERSION_LESS "3.0.0" )
  target_compile_definitions( viame_opencv PRIVATE -DVIAME_OPENCV_VER_2 )
endif()

get_target_property( results viame_opencv COMPILE_DEFINITIONS )

target_link_libraries( viame_opencv
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
                       kwiver::kwiversys
                       viame_opencv_bridge
                       ${OpenCV_LIBRARIES}
                       opencv_ml
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

algorithms_create_plugin( viame_opencv
  register_algorithms.cxx
  )

target_link_libraries( viame_opencv_plugin
  PUBLIC               kwiver::vital_vpm
  )

# Add process library
set( process_headers
  detect_in_subregions_process.h
  measure_objects_process.h
  pair_stereo_detections_process.h
  pair_stereo_tracks_process.h
)

set( process_sources
  register_processes.cxx
  detect_in_subregions_process.cxx
  measure_objects_process.cxx
  pair_stereo_detections_process.cxx
  pair_stereo_tracks_process.cxx
)

kwiver_add_plugin( viame_processes_opencv
  SUBDIR           ${viame_plugin_process_subdir}
  SOURCES          ${process_sources}
  PRIVATE          kwiver::vital
                   kwiver::vital_algo
                   kwiver::vital_config
                   kwiver::vital_logger
                   kwiver::vital_util
                   kwiver::vital_vpm
                   kwiver::sprokit_pipeline
                   viame_opencv_bridge
                   viame_measurement
                   ${OpenCV_LIBRARIES}
                   viame_opencv
)

target_include_directories( viame_processes_opencv PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/plugins>
  )

if( VIAME_ENABLE_PYTHON )
  set( _rel_modnames
    __init__
    fft_filter_based_on_ref
    multimodal_registration
    prior_coverage_opencv
    registration_utils
    stereo_algos
    stereo_demo
    stereo_expt
    stereo_pipeline
    stereo_processes
    stereo_utils
    watershed_segmenter
  )

  foreach( rel_modname IN ITEMS ${_rel_modnames} )
    string( REPLACE "." "/" rel_modpath_noext ${rel_modname} )
    kwiver_add_python_module(
      "${CMAKE_CURRENT_SOURCE_DIR}/${rel_modpath_noext}.py"
      "opencv"
      "${rel_modname}" )
  endforeach()
endif()

