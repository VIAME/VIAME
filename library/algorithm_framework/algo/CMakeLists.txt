###
# Build file for abstract algorithms
#

#
set( public_headers
  algorithm.h
  algorithm.txx
  associate_detections_to_tracks.h
  close_loops.h
  compute_ref_homography.h
  compute_stereo_depth_map.h
  compute_track_descriptors.h
  detect_features.h
  detect_motion.h
  detected_object_filter.h
  detected_object_set_input.h
  detected_object_set_output.h
  draw_detected_object_set.h
  estimate_fundamental_matrix.h
  estimate_homography.h
  extract_descriptors.h
  feature_descriptor_io.h
  filter_features.h
  handle_descriptor_request.h
  image_filter.h
  image_io.h
  image_object_detector.h
  initialize_object_tracks.h
  match_descriptor_sets.h
  match_features.h
  merge_detections.h
  merge_images.h
  optimize_cameras.h
  perform_text_query.h
  query_track_descriptor_set.h
  read_object_track_set.h
  read_track_descriptor_set.h
  refine_detections.h
  refine_tracks.h
  resection_camera.h
  segment_via_points.h
  split_image.h
  track_features.h
  track_objects.h
  train_detector.h
  train_tracker.h
  transform_2d_io.h
  video_input.h
  video_output.h
  warp_image.h
  write_object_track_set.h
  write_track_descriptor_set.h
  )

set( sources
  algorithm.cxx
  associate_detections_to_tracks.cxx
  close_loops.cxx
  compute_ref_homography.cxx
  compute_stereo_depth_map.cxx
  compute_track_descriptors.cxx
  detect_features.cxx
  detect_motion.cxx
  detected_object_filter.cxx
  detected_object_set_input.cxx
  detected_object_set_output.cxx
  draw_detected_object_set.cxx
  estimate_fundamental_matrix.cxx
  estimate_homography.cxx
  extract_descriptors.cxx
  feature_descriptor_io.cxx
  filter_features.cxx
  handle_descriptor_request.cxx
  image_filter.cxx
  image_io.cxx
  image_object_detector.cxx
  initialize_object_tracks.cxx
  match_descriptor_sets.cxx
  match_features.cxx
  merge_detections.cxx
  merge_images.cxx
  optimize_cameras.cxx
  perform_text_query.cxx
  query_track_descriptor_set.cxx
  read_object_track_set.cxx
  read_track_descriptor_set.cxx
  refine_detections.cxx
  refine_tracks.cxx
  resection_camera.cxx
  segment_via_points.cxx
  split_image.cxx
  track_features.cxx
  track_objects.cxx
  train_detector.cxx
  train_tracker.cxx
  transform_2d_io.cxx
  video_input.cxx
  video_output.cxx
  warp_image.cxx
  write_object_track_set.cxx
  write_track_descriptor_set.cxx
)

###
## Install header files


# install export header
kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_algo_export.h
  SUBDIR   viame/algorithm_framework/algo
  NOPATH
  )

kwiver_add_library( vital_algo SHARED
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
                  vital_types
                  vital_vpm
  )

# Needs to be updated for new API
#if (KWIVER_ENABLE_TOOLS)
####
##     plugins for plugin explorer
#kwiver_add_plugin( algo_explorer_plugin
#  SUBDIR   ${kwiver_plugin_plugin_explorer_subdir}
#  SOURCES  algo_explorer_plugin.cxx
#  PRIVATE  vital
#           vital_vpm
#           vital_config
#           vital_algo
#           explorer_plugin )
#endif()

# Republished under the prefix the imported sources use.
viame_lite_generated( vital_algo_export.h algorithm_framework/algo )
