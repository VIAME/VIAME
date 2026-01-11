
###
# Loadable algorithms
##

find_package( OpenCV REQUIRED )

include_directories( "${OpenCV_INCLUDE_DIRS}" )
link_directories( "${OpenCV_LIBRARY_DIR}" )

set( plugin_headers
  apply_color_correction.h
  enhance_images.h
  debayer_filter.h
  random_hue_shift.h
  calibrate_single_camera.h
  calibrate_stereo_cameras.h
  compute_stereo_disparity.h
  detect_calibration_targets.h
  optimize_stereo_cameras.h
  filter_stereo_feature_tracks.h
  kmedians.h
  add_keypoints_from_mask.h
  split_image_habcam.h
  split_image_horizontally.h
  pair_stereo_detections.h
  pair_stereo_tracks.h
  refine_detections_grabcut.h
  refine_detections_util.h
  refine_detections_watershed.h
  windowed_detector.h
  windowed_refiner.h
  windowed_trainer.h
  windowed_utils.h
  ellipse_proposal.h
  diff_of_gauss_detector.h
  gabor_features.h
  hog_features.h
  canny_edge_detector.h
  adaboost_classifier.h
  classify_fish_hierarchical_svm.h
  )

set( plugin_sources
  apply_color_correction.cxx
  enhance_images.cxx
  debayer_filter.cxx
  random_hue_shift.cxx
  calibrate_single_camera.cxx
  calibrate_stereo_cameras.cxx
  compute_stereo_disparity.cxx
  detect_calibration_targets.cxx
  optimize_stereo_cameras.cxx
  filter_stereo_feature_tracks.cxx
  kmedians.cxx
  add_keypoints_from_mask.cxx
  split_image_habcam.cxx
  split_image_horizontally.cxx
  pair_stereo_detections.cxx
  pair_stereo_tracks.cxx
  refine_detections_grabcut.cxx
  refine_detections_util.cxx
  refine_detections_watershed.cxx
  windowed_detector.cxx
  windowed_refiner.cxx
  windowed_trainer.cxx
  windowed_utils.cxx
  diff_of_gauss_detector.cxx
  gabor_features.cxx
  hog_features.cxx
  canny_edge_detector.cxx
  adaboost_classifier.cxx
  classify_fish_hierarchical_svm.cxx
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
                       opencv_ml
                       viame_core
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

target_include_directories( viame_processes_opencv PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
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

