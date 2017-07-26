###
# Build file for abstract algorithms
#

# ==================================================================
###
# These headers are installed
#
set( public_headers

  algorithm.h
  algorithm_factory.h
  analyze_tracks.h
  associate_detections_to_tracks.h
  bundle_adjust.h
  close_loops.h
  compute_association_matrix.h
  compute_ref_homography.h
  compute_stereo_depth_map.h
  compute_track_descriptors.h
  convert_image.h
  detect_features.h
  detected_object_filter.h
  detected_object_set_input.h
  detected_object_set_output.h
  draw_detected_object_set.h
  draw_tracks.h
  dynamic_configuration.h
  estimate_canonical_transform.h
  estimate_essential_matrix.h
  estimate_fundamental_matrix.h
  estimate_homography.h
  estimate_similarity_transform.h
  extract_descriptors.h
  feature_descriptor_io.h
  filter_features.h
  formulate_query.h
  filter_tracks.h
  geo_map.h
  image_filter.h
  image_io.h
  image_object_detector.h
  initialize_cameras_landmarks.h
  initialize_object_tracks.h
  match_features.h
  optimize_cameras.h
  refine_detections.h
  split_image.h
  track_features.h
  train_detector.h
  track_descriptor_set_input.h
  track_descriptor_set_output.h
  triangulate_landmarks.h
  uuid_factory.h
  video_input.h
  )

set( sources

  algorithm.cxx
  algorithm_factory.cxx
  analyze_tracks.cxx
  associate_detections_to_tracks.cxx
  bundle_adjust.cxx
  close_loops.cxx
  compute_association_matrix.cxx
  compute_ref_homography.cxx
  compute_stereo_depth_map.cxx
  compute_track_descriptors.cxx
  convert_image.cxx
  detect_features.cxx
  detected_object_filter.cxx
  detected_object_set_input.cxx
  detected_object_set_output.cxx
  draw_detected_object_set.cxx
  draw_tracks.cxx
  dynamic_configuration.cxx
  estimate_canonical_transform.cxx
  estimate_essential_matrix.cxx
  estimate_fundamental_matrix.cxx
  estimate_homography.cxx
  estimate_similarity_transform.cxx
  extract_descriptors.cxx
  feature_descriptor_io.cxx
  filter_features.cxx
  formulate_query.cxx
  filter_tracks.cxx
  geo_map.cxx
  image_filter.cxx
  image_io.cxx
  image_object_detector.cxx
  initialize_object_tracks.cxx
  initialize_cameras_landmarks.cxx
  match_features.cxx
  optimize_cameras.cxx
  refine_detections.cxx
  split_image.cxx
  track_features.cxx
  train_detector.cxx
  track_descriptor_set_input.cxx
  track_descriptor_set_output.cxx
  triangulate_landmarks.cxx
  uuid_factory.cxx
  video_input.cxx
)

kwiver_install_headers(
  ${public_headers}
  SUBDIR   vital/algo
  )

# install export header
kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_algo_export.h
  SUBDIR   vital/algo
  NOPATH
  )

kwiver_add_library( vital_algo
  ${public_headers}
  ${sources}
  ${CMAKE_CURRENT_BINARY_DIR}/vital_algo_export.h
  )

target_link_libraries( vital_algo
  PRIVATE         kwiversys
                  vital_util
  PUBLIC          vital
                  vital_config
                  vital_logger
                  vital_vpm
  )

if (KWIVER_ENABLE_TOOLS)
###
#     plugins for plugin explorer
kwiver_add_plugin( algo_explorer_plugin
  SUBDIR   modules/plugin_explorer
  SOURCES  algo_explorer_plugin.cxx
  PRIVATE  vital
           vital_vpm
           vital_config
           vital_algo
           explorer_plugin )
endif()
