# Build / Install Plugin containing core algorithm implementations

set( plugin_core_headers
  associate_detections_to_tracks_threshold.h
  average_track_descriptors.h
  class_probablity_filter.h
  close_loops_appearance_indexed.h
  close_loops_bad_frames_only.h
  close_loops_exhaustive.h
  close_loops_keyframe.h
  close_loops_multi_method.h
  colorize.h
  compute_association_matrix_from_features.h
  compute_ref_homography_core.h
  convert_image_bypass.h
  create_detection_grid.h
  depth_utils.h
  detect_features_filtered.h
  detected_object_set_input_kw18.h
  detected_object_set_output_kw18.h
  detected_object_set_input_csv.h
  detected_object_set_output_csv.h
  detected_object_set_input_simulator.h
  dynamic_config_none.h
  estimate_canonical_transform.h
  example_detector.h
  feature_descriptor_io.h
  filter_features_magnitude.h
  filter_features_nonmax.h
  filter_features_scale.h
  filter_tracks.h
  full_frame_detector.h
  handle_descriptor_request_core.h
  hierarchical_bundle_adjust.h
  initialize_cameras_landmarks.h
  initialize_cameras_landmarks_keyframe.h
  initialize_object_tracks_threshold.h
  keyframe_selector_basic.h
  match_features_fundamental_matrix.h
  match_features_homography.h
  match_tracks.h
  mesh_operations.h
  necker_reverse.h
  read_object_track_set_kw18.h
  read_track_descriptor_set_csv.h
  refine_detections_add_fixed.h
  refine_detections_nms.h
  render_mesh_depth_map.h
  track_features_augment_keyframes.h
  track_features_core.h
  transfer_bbox_with_depth_map.h
  transform.h
  transform_detected_object_set.h
  triangle_scan_iterator.h
  triangulate_landmarks.h
  uv_unwrap_mesh.h
  video_input_filter.h
  video_input_image_list.h
  video_input_pos.h
  video_input_splice.h
  video_input_split.h
  write_object_track_set_kw18.h
  write_track_descriptor_set_csv.h

  epipolar_geometry.h
  interpolate_camera.h
  interpolate_track_spline.h
  match_features_homography.h
  match_matrix.h
  metrics.h
  projected_track_set.h
  sfm_utils.h
  track_features_core.h
  track_set_impl.h
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
  average_track_descriptors.cxx
  class_probablity_filter.cxx
  close_loops_appearance_indexed.cxx
  close_loops_bad_frames_only.cxx
  close_loops_exhaustive.cxx
  close_loops_keyframe.cxx
  close_loops_multi_method.cxx
  colorize.cxx
  compute_association_matrix_from_features.cxx
  compute_ref_homography_core.cxx
  convert_image_bypass.cxx
  create_detection_grid.cxx
  depth_utils.cxx
  detect_features_filtered.cxx
  detected_object_set_input_kw18.cxx
  detected_object_set_output_kw18.cxx
  detected_object_set_input_csv.cxx
  detected_object_set_output_csv.cxx
  detected_object_set_input_simulator.cxx
  dynamic_config_none.cxx
  estimate_canonical_transform.cxx
  example_detector.cxx
  feature_descriptor_io.cxx
  filter_features_magnitude.cxx
  filter_features_nonmax.cxx
  filter_features_scale.cxx
  filter_tracks.cxx
  full_frame_detector.cxx
  handle_descriptor_request_core.cxx
  hierarchical_bundle_adjust.cxx
  initialize_cameras_landmarks.cxx
  initialize_cameras_landmarks_keyframe.cxx
  initialize_object_tracks_threshold.cxx
  keyframe_selector_basic.cxx
  match_features_fundamental_matrix.cxx
  match_features_homography.cxx
  match_tracks.cxx
  necker_reverse.cxx
  mesh_operations.cxx
  read_object_track_set_kw18.cxx
  read_track_descriptor_set_csv.cxx
  refine_detections_add_fixed.cxx
  refine_detections_nms.cxx
  render_mesh_depth_map.cxx
  track_features_augment_keyframes.cxx
  track_features_core.cxx
  transfer_bbox_with_depth_map.cxx
  transform.cxx
  transform_detected_object_set.cxx
  triangle_scan_iterator.cxx
  triangulate_landmarks.cxx
  uv_unwrap_mesh.cxx
  video_input_filter.cxx
  video_input_image_list.cxx
  video_input_pos.cxx
  video_input_splice.cxx
  video_input_split.cxx
  write_object_track_set_kw18.cxx
  write_track_descriptor_set_csv.cxx

  epipolar_geometry.cxx
  interpolate_camera.cxx
  interpolate_track_spline.cxx
  match_features_homography.cxx
  match_matrix.cxx
  metrics.cxx
  projected_track_set.cxx
  sfm_utils.cxx
  track_features_core.cxx
  track_set_impl.cxx
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
                       vital_config
  )

algorithms_create_plugin( kwiver_algo_core
  register_algorithms.cxx
  )

if (KWIVER_ENABLE_TESTS)
  add_subdirectory(tests)
endif()

if (KWIVER_ENABLE_TOOLS)
  add_subdirectory(applets)
endif()
