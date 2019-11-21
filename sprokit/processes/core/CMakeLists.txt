#
# KWIVER processes
#
project( kwiver_processes )

set( sources
  register_processes.cxx

  associate_detections_to_tracks_process.cxx
  close_loops_process.cxx
  compute_association_matrix_process.cxx
  compute_homography_process.cxx
  compute_stereo_depth_map_process.cxx
  compute_track_descriptors_process.cxx
  convert_tracks_to_detections_process.cxx
  detect_features_if_keyframe_process.cxx
  detect_features_process.cxx
  detect_motion_process.cxx
  detected_object_output_process.cxx
  detected_object_input_process.cxx
  detected_object_filter_process.cxx
  downsample_process.cxx
  draw_detected_object_set_process.cxx
  draw_tracks_process.cxx
  extract_descriptors_process.cxx
  frame_list_process.cxx
  handle_descriptor_request_process.cxx
  image_file_reader_process.cxx
  image_filter_process.cxx
  image_object_detector_process.cxx
  image_writer_process.cxx
  initialize_object_tracks_process.cxx
  keyframe_selection_process.cxx
  matcher_process.cxx
  merge_detection_sets_process.cxx
  merge_images_process.cxx
  merge_track_sets_process.cxx
  perform_query_process.cxx
  print_config_process.cxx
  read_descriptor_process.cxx
  read_object_track_process.cxx
  read_track_descriptor_process.cxx
  refine_detections_process.cxx
  serializer_base.cxx
  serializer_process.cxx
  deserializer_process.cxx
  shift_detected_object_set_frames_process.cxx
  split_image_process.cxx
  stabilize_image_process.cxx
  track_features_process.cxx
  train_detector_process.cxx
  unwrap_detections_process.cxx
  video_input_process.cxx
  write_object_track_process.cxx
  write_track_descriptor_process.cxx
)

set( private_headers
  associate_detections_to_tracks_process.h
  close_loops_process.h
  compute_association_matrix_process.h
  compute_homography_process.h
  compute_stereo_depth_map_process.h
  compute_track_descriptors_process.h
  convert_tracks_to_detections_process.h
  detect_features_if_keyframe_process.h
  detect_features_process.h
  detected_object_output_process.h
  detected_object_input_process.h
  detected_object_filter_process.h
  detect_motion_process.h
  downsample_process.h
  draw_detected_object_set_process.h
  draw_tracks_process.h
  extract_descriptors_process.h
  frame_list_process.h
  handle_descriptor_request_process.h
  image_file_reader_process.h
  image_filter_process.h
  image_object_detector_process.h
  image_writer_process.h
  initialize_object_tracks_process.h
  keyframe_selection_process.h
  matcher_process.h
  merge_detection_sets_process.h
  merge_images_process.h
  merge_track_sets_process.h
  perform_query_process.h
  print_config_process.h
  read_descriptor_process.h
  read_object_track_process.h
  read_track_descriptor_process.h
  refine_detections_process.h
  serializer_base.h
  serializer_process.h
  deserializer_process.h
  shift_detected_object_set_frames_process.h
  split_image_process.h
  stabilize_image_process.h
  track_features_process.h
  train_detector_process.h
  unwrap_detections_process.h
  video_input_process.h
  write_object_track_process.h
  write_track_descriptor_process.h
)

kwiver_private_header_group( ${private_headers} )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

kwiver_add_plugin( kwiver_processes
  SUBDIR           ${kwiver_plugin_process_subdir}
  SOURCES          ${sources}
                   ${private_headers}
  PRIVATE          sprokit_pipeline
                   kwiver_algo_core
                   kwiver_adapter
                   kwiversys
                   vital vital_vpm vital_logger vital_config
                   ${Boost_SYSTEM_LIBRARY}
                   ${Boost_FILESYSTEM_LIBRARY}
)
