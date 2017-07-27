# Build / Install Plugin containing core algorithm implementations

set( plugin_core_headers
  associate_detections_to_tracks_threshold.h
  class_probablity_filter.h
  close_loops_bad_frames_only.h
  close_loops_exhaustive.h
  close_loops_keyframe.h
  close_loops_multi_method.h
  compute_association_matrix_from_features.h
  compute_ref_homography_core.h
  convert_image_bypass.h
  detected_object_set_input_kw18.h
  detected_object_set_output_kw18.h
  detected_object_set_input_csv.h
  detected_object_set_output_csv.h
  dynamic_config_none.h
  estimate_canonical_transform.h
  feature_descriptor_io.h
  filter_features_magnitude.h
  formulate_query_core.h
  filter_features_scale.h
  filter_tracks.h
  hierarchical_bundle_adjust.h
  initialize_cameras_landmarks.h
  initialize_object_tracks_threshold.h
  match_features_fundamental_matrix.h
  match_features_homography.h
  track_descriptor_set_output_csv.h
  merge_tracks.h
  track_features_core.h
  transform.h
  triangulate_landmarks.h
  video_input_filter.h
  video_input_image_list.h
  video_input_pos.h
  video_input_split.h

  epipolar_geometry.h
  interpolate_camera.h
  match_features_homography.h
  match_matrix.h
  metrics.h
  projected_track_set.h
  track_features_core.h
  transform.h
  triangulate.h
  triangulate_landmarks.h
  )

kwiver_install_headers(
  SUBDIR     arrows/core
  ${plugin_core_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_core_export.h
  NOPATH   SUBDIR     arrows/core
  )

set( plugin_core_sources
  associate_detections_to_tracks_threshold.cxx
  class_probablity_filter.cxx
  close_loops_bad_frames_only.cxx
  close_loops_exhaustive.cxx
  close_loops_keyframe.cxx
  close_loops_multi_method.cxx
  compute_association_matrix_from_features.cxx
  compute_ref_homography_core.cxx
  convert_image_bypass.cxx
  detected_object_set_input_kw18.cxx
  detected_object_set_output_kw18.cxx
  detected_object_set_input_csv.cxx
  detected_object_set_output_csv.cxx
  dynamic_config_none.cxx
  estimate_canonical_transform.cxx
  feature_descriptor_io.cxx
  filter_features_magnitude.cxx
  formulate_query_core.cxx
  filter_features_scale.cxx
  filter_tracks.cxx
  hierarchical_bundle_adjust.cxx
  initialize_cameras_landmarks.cxx
  initialize_object_tracks_threshold.cxx
  match_features_fundamental_matrix.cxx
  match_features_homography.cxx
  track_descriptor_set_output_csv.cxx
  merge_tracks.cxx
  track_features_core.cxx
  transform.cxx
  triangulate_landmarks.cxx
  video_input_filter.cxx
  video_input_image_list.cxx
  video_input_pos.cxx
  video_input_split.cxx

  epipolar_geometry.cxx
  interpolate_camera.cxx
  match_features_homography.cxx
  match_matrix.cxx
  metrics.cxx
  projected_track_set.cxx
  track_features_core.cxx
  transform.cxx
  triangulate.cxx
  triangulate_landmarks.cxx
  )

# Address MSVC fatal error C1128:
# number of sections exceeded object file format limit
if(MSVC)
  set_source_files_properties(
    triangulate.cxx
    PROPERTIES COMPILE_FLAGS "/bigobj"
    )
endif()

kwiver_add_library( kwiver_algo_core
  ${plugin_core_headers}
  ${plugin_core_sources}
  )

target_link_libraries( kwiver_algo_core
  PUBLIC               vital_algo
  PRIVATE              kwiversys
                       vital_video_metadata
                       vital_config
  )

algorithms_create_plugin( kwiver_algo_core
  register_algorithms.cxx
  )

if (KWIVER_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
