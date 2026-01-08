# Build / Install Plugin containing core algorithm implementations

set( plugin_headers
  add_timestamp_from_filename.h
  auto_detect_transform.h
  average_track_descriptors.h
  convert_head_tail_points.h
  convert_notes_to_attributes.h
  convert_polygons_to_mask.h
  detect_shot_breaks.h
  empty_detector.h
  filename_to_timestamp.h
  full_frame_detector.h
  measurement_utilities.h
  merge_detections_suppress_in_regions.h
  pair_stereo_detections.h
  read_detected_object_set_fishnet.h
  read_detected_object_set_habcam.h
  read_detected_object_set_oceaneyes.h
  read_detected_object_set_viame_csv.h
  read_object_track_set_viame_csv.h
  refine_detections_add_fixed.h
  refine_detections_nms.h
  windowed_detector.h
  windowed_refiner.h
  windowed_trainer.h
  windowed_utils.h
  write_detected_object_set_viame_csv.h
  write_disparity_maps.h
  write_object_track_set_viame_csv.h
  )

set( plugin_sources
  add_timestamp_from_filename.cxx
  auto_detect_transform.cxx
  average_track_descriptors.cxx
  convert_head_tail_points.cxx
  convert_notes_to_attributes.cxx
  convert_polygons_to_mask.cxx
  detect_shot_breaks.cxx
  empty_detector.cxx
  filename_to_timestamp.cxx
  full_frame_detector.cxx
  measurement_utilities.cxx
  merge_detections_suppress_in_regions.cxx
  pair_stereo_detections.cxx
  read_detected_object_set_fishnet.cxx
  read_detected_object_set_habcam.cxx
  read_detected_object_set_oceaneyes.cxx
  read_detected_object_set_viame_csv.cxx
  read_object_track_set_viame_csv.cxx
  refine_detections_add_fixed.cxx
  refine_detections_nms.cxx
  windowed_detector.cxx
  windowed_refiner.cxx
  windowed_trainer.cxx
  windowed_utils.cxx
  write_detected_object_set_viame_csv.cxx
  write_disparity_maps.cxx
  write_object_track_set_viame_csv.cxx
  )

kwiver_install_headers(
  SUBDIR     viame
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_core_export.h
  NOPATH   SUBDIR     viame
  )

kwiver_add_library( viame_core
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

find_package( Boost ${KWIVER_BOOST_VERSION} REQUIRED
  COMPONENTS
  chrono
  filesystem
  system)

include_directories( SYSTEM ${Boost_INCLUDE_DIRS} )
link_directories( ${Boost_LIBRARY_DIRS} )

set( CORE_LINK_LIBRARIES
  kwiver::vital
  kwiver::vital_algo
  kwiver::vital_config
  kwiver::vital_exceptions
  kwiver::vital_logger
  kwiver::vital_util
  kwiver::kwiversys
  kwiver::kwiver_algo_mvg
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  )

# Eigen is used by metadata parsers
find_package( Eigen3 REQUIRED NO_MODULE )

# VXL is used in some parsers for aux functionality if enabled
if( VIAME_ENABLE_VXL )
  find_package( VXL REQUIRED )
  include( ${VXL_CMAKE_DIR}/UseVXL.cmake )
  include_directories( SYSTEM ${VXL_CORE_INCLUDE_DIR} )
  include_directories( SYSTEM ${VXL_VGL_INCLUDE_DIR} )
  link_directories( ${VXL_LIBRARY_DIR} )
  add_definitions( -DVIAME_ENABLE_VXL )
  set( AUX_LINK_LIBRARIES vgl ${AUX_LINK_LIBRARIES} )
endif()

# OpenCV is used in some parsers for aux functionality if enabled
if( VIAME_ENABLE_OPENCV )
  find_package( OpenCV REQUIRED )
  include_directories( "${OpenCV_INCLUDE_DIRS}" )
  link_directories( "${OpenCV_LIBRARY_DIR}" )
  add_definitions( -DVIAME_ENABLE_OPENCV )
  set( AUX_LINK_LIBRARIES
       ${OpenCV_LIBRARIES}
       kwiver::kwiver_algo_ocv
       ${AUX_LINK_LIBRARIES} )
endif()

target_link_libraries( viame_core
  PUBLIC               ${CORE_LINK_LIBRARIES}
                       ${AUX_LINK_LIBRARIES}
  )

set_target_properties( viame_core PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

algorithms_create_plugin( viame_core
  register_algorithms.cxx
  )

target_link_libraries( viame_core_plugin
  PUBLIC               kwiver::vital_vpm
  )

find_package( Boost ${KWIVER_BOOST_VERSION} REQUIRED
    COMPONENTS
      chrono
      filesystem
      system)

include_directories( SYSTEM ${Boost_INCLUDE_DIRS} )
link_directories( ${Boost_LIBRARY_DIRS} )

# Add process library
set( process_headers
  accumulate_image_statistics_process.h
  accumulate_object_tracks_process.h
  align_multimodal_imagery_process.h
  calibrate_cameras_from_tracks_process.h
  create_database_query_process.h
  select_database_query_process.h
  detect_shot_breaks_process.h
  extract_desc_ids_for_training_process.h
  fetch_descriptors_process.h
  filter_frame_index_process.h
  filter_frame_process.h
  image_to_image_set_process.h
  filter_object_tracks_process.h
  ingest_descriptors_process.h
  measure_objects_process.h
  object_track_descriptors_process.h
  pair_stereo_detections_process.h
  write_query_results_as_tracks_process.h
  read_habcam_metadata_process.h
  refine_measurements_process.h
  split_tracks_to_feature_landmarks_process.h
  stack_frames_process.h
  store_descriptors_csv.h
  track_conductor_process.h
  write_homography_list_process.h
)

set( process_sources
  register_processes.cxx
  accumulate_image_statistics_process.cxx
  accumulate_object_tracks_process.cxx
  align_multimodal_imagery_process.cxx
  calibrate_cameras_from_tracks_process.cxx
  create_database_query_process.cxx
  select_database_query_process.cxx
  detect_shot_breaks_process.cxx
  extract_desc_ids_for_training_process.cxx
  fetch_descriptors_process.cxx
  filter_frame_index_process.cxx
  filter_frame_process.cxx
  image_to_image_set_process.cxx
  filter_object_tracks_process.cxx
  ingest_descriptors_process.cxx
  measure_objects_process.cxx
  object_track_descriptors_process.cxx
  pair_stereo_detections_process.cxx
  write_query_results_as_tracks_process.cxx
  read_habcam_metadata_process.cxx
  refine_measurements_process.cxx
  split_tracks_to_feature_landmarks_process.cxx
  stack_frames_process.cxx
  store_descriptors_csv.cxx
  track_conductor_process.cxx
  write_homography_list_process.cxx
)

kwiver_add_plugin( viame_processes_core
  SUBDIR           ${viame_plugin_process_subdir}
  SOURCES          ${process_sources}
                   ${private_headers}
  PRIVATE          ${CORE_LINK_LIBRARIES}
                   kwiver::sprokit_pipeline
                   kwiver::kwiver_algo_core
                   kwiver::kwiver_adapter
                   kwiver::vital_util
                   kwiver::vital_vpm
                   Eigen3::Eigen
                   kwiver::kwiver_algo_mvg
                   kwiver::kwiver_algo_ocv
                   ${Boost_SYSTEM_LIBRARY}
                   ${Boost_FILESYSTEM_LIBRARY}
                   viame_core
                   viame_opencv
 )

target_include_directories( viame_processes_core PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

if( VIAME_ENABLE_PYTHON )

  kwiver_create_python_init( arrows/core )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/normalize_image_npy.py
    arrows/core
    normalize_image_npy )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/merge_detections_nms_fusion.py
    arrows/core
    merge_detections_nms_fusion )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    processes/core
    __init__ )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/utility_processes.py
    processes/core
    utility_processes )

endif()

