
###
# Sub folder options
##
if( VIAME_ENABLE_UW_PREDICTOR )
  add_subdirectory( uw_predictor )
endif()

###
# Loadable algorithms
##

find_package( OpenCV REQUIRED )

include_directories( "${OpenCV_INCLUDE_DIRS}" )
link_directories( "${OpenCV_LIBRARY_DIR}" )

set( plugin_headers
  image_enhancement.h
  debayer_filter.h
  random_hue_shift.h
  stereo_calibration.h
  stereo_disparity_map.h
  detect_calibration_targets.h
  optimize_stereo_cameras.h
  stereo_feature_track_filter.h
  kmedians.h
  keypoints_from_mask.h
  add_keypoints_from_mask.h
  split_image_habcam.h
  pair_stereo_detections.h
  pair_stereo_tracks.h
  )

set( plugin_sources
  image_enhancement.cxx
  debayer_filter.cxx
  random_hue_shift.cxx
  stereo_calibration.cxx
  stereo_disparity_map.cxx
  detect_calibration_targets.cxx
  optimize_stereo_cameras.cxx
  stereo_feature_track_filter.cxx
  kmedians.cxx
  keypoints_from_mask.cxx
  add_keypoints_from_mask.cxx
  split_image_habcam.cxx
  pair_stereo_detections.cxx
  pair_stereo_tracks.cxx
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
                       kwiver::kwiver_algo_ocv
                       ${OpenCV_LIBRARIES}
  )

target_include_directories(viame_opencv PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR})

set_target_properties( viame_opencv PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

algorithms_create_plugin( viame_opencv
  register_algorithms.cxx
  )

target_link_libraries( viame_opencv_plugin
  PUBLIC               kwiver::vital_vpm
  )

# Eigen is required for MVG triangulation
find_package( Eigen3 REQUIRED NO_MODULE )

# Add process library
set( process_headers
  measure_objects_process.h
  calibrate_single_camera_process.h
  pair_stereo_detections_process.h
  pair_stereo_tracks_process.h
)

set( process_sources
  register_processes.cxx
  measure_objects_process.cxx
  calibrate_single_camera_process.cxx
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
                   kwiver::kwiver_algo_ocv
                   kwiver::kwiver_algo_mvg
                   Eigen3::Eigen
                   ${OpenCV_LIBRARIES}
                   viame_opencv
)

if( VIAME_ENABLE_PYTHON )
  set( _rel_modnames
    __init__
    multimodal_registration
    fft_filter_based_on_ref
    stereo_algos
    stereo_demo
    stereo_expt
    stereo_pipeline
    stereo_processes
    stereo_utils
    )

  foreach( rel_modname IN ITEMS ${_rel_modnames} )
    string( REPLACE "." "/" rel_modpath_noext ${rel_modname} )
    kwiver_add_python_module(
      "${CMAKE_CURRENT_SOURCE_DIR}/${rel_modpath_noext}.py"
      "processes/opencv"
      "${rel_modname}" )
  endforeach()
endif()

if (VIAME_BUILD_TESTS)
  add_subdirectory(tests)
endif()