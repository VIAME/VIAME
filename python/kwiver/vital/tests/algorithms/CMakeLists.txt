set(__test_algorithms
  simple_associate_detections_to_tracks
  simple_bundle_adjust
  simple_close_loops
  simple_image_object_detector
  simple_initialize_cameras_landmarks
  simple_initialize_object_tracks
  simple_integrate_depth_maps
  simple_interpolate_track
  simple_keyframe_selection
  simple_match_descriptor_sets
  simple_match_features
  simple_merge_images
  simple_optimize_cameras
  simple_read_object_track_set
  simple_read_track_descriptor_set
  simple_refine_detections
  simple_split_image
  simple_track_features
  simple_train_detector
  simple_transform_2d_io
  simple_triangulate_landmarks
  simple_uuid_factory
  simple_video_input
  simple_uv_unwrap_mesh
  simple_write_object_track_set
  simple_write_track_descriptor_set
  simple_associate_detections_to_tracks
  simple_bundle_adjust
  )

set(test_algorithms
    __init__
    ${__test_algorithms}
   )

foreach (modname IN LISTS test_algorithms)
  kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/${modname}.py
    vital/tests/algorithms
    ${modname})
endforeach()
