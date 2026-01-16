
###
# Loadable algorithms
##

find_package( OpenCV REQUIRED )

include_directories( "${OpenCV_INCLUDE_DIRS}" )
link_directories( "${OpenCV_LIBRARY_DIR}" )

set( plugin_headers
  adaboost_classifier.h
  add_keypoints_from_mask.h
  apply_color_correction.h
  calibrate_single_camera.h
  calibrate_stereo_cameras.h
  canny_edge_detector.h
  classify_fish_hierarchical_svm.h
  compute_stereo_disparity.h
  debayer_filter.h
  detect_calibration_targets.h
  diff_of_gauss_detector.h
  ellipse_proposal.h
  enhance_images.h
  filter_stereo_feature_tracks.h
  gabor_features.h
  hog_features.h
  kmedians.h
  optimize_stereo_cameras.h
  pair_stereo_detections.h
  pair_stereo_tracks.h
  plot_metrics.h
  random_hue_shift.h
  refine_detections_grabcut.h
  refine_detections_util.h
  refine_detections_watershed.h
  split_image_habcam.h
  split_image_horizontally.h
  windowed_detector.h
  windowed_refiner.h
  windowed_trainer.h
  windowed_utils.h
)

set( plugin_sources
  adaboost_classifier.cxx
  add_keypoints_from_mask.cxx
  apply_color_correction.cxx
  calibrate_single_camera.cxx
  calibrate_stereo_cameras.cxx
  canny_edge_detector.cxx
  classify_fish_hierarchical_svm.cxx
  compute_stereo_disparity.cxx
  debayer_filter.cxx
  detect_calibration_targets.cxx
  diff_of_gauss_detector.cxx
  enhance_images.cxx
  filter_stereo_feature_tracks.cxx
  gabor_features.cxx
  hog_features.cxx
  kmedians.cxx
  optimize_stereo_cameras.cxx
  pair_stereo_detections.cxx
  pair_stereo_tracks.cxx
  plot_metrics.cxx
  random_hue_shift.cxx
  refine_detections_grabcut.cxx
  refine_detections_util.cxx
  refine_detections_watershed.cxx
  split_image_habcam.cxx
  split_image_horizontally.cxx
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
  calibrate_single_camera_process.h
  detect_in_subregions_process.h
  measure_objects_process.h
  pair_stereo_detections_process.h
  pair_stereo_tracks_process.h
)

set( process_sources
  register_processes.cxx
  calibrate_single_camera_process.cxx
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
    fft_filter_based_on_ref
    multimodal_registration
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
      "opencv"
      "${rel_modname}" )
  endforeach()
endif()

