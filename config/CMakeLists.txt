set(CMAKE_FOLDER "Configuration")

set(config_files
  ceres_bundle_adjuster.conf
  core_feature_descriptor_io.conf
  core_feature_tracker.conf
  core_fmatrix_guided_feature_matcher.conf
  core_homography_guided_feature_matcher.conf
  core_image_list_video_input.conf
  core_nonmax_detector.conf
  core_output_homography_generator.conf
  core_pca_canonical_tfm.conf
  core_track_filter.conf
  core_video_input_image_list.conf
  core_video_input_image_pos_list.conf
  cuda_integrate_depth_maps.conf
  default_camera_intrinsics.conf
  ffmpeg_video_input.conf
  loop_closer_bad_frames.conf
  loop_closer_homography.conf
  loop_closer_keyframe.conf
  loop_closer_multi_method.conf
  mvg_hierarchical_bundle_adjuster.conf
  mvg_initialize_cameras_landmarks.conf
  mvg_initialize_cameras_landmarks_basic.conf
  ocv_klt_tracker.conf
  ocv_ORB_detector_descriptor.conf
  ocv_SURF_detector_descriptor.conf
  super3d_depth_map.conf)

# Ensure target directory exists
set(out_dir "${KWIVER_BINARY_DIR}/${kwiver_config_subdir}")
add_custom_command(
  OUTPUT "${out_dir}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${out_dir}"
  )

# Set up meta-target
add_custom_target(kwiver-data-config ALL DEPENDS "${out_dir}")

# Add targets to copy config files to the build area
foreach(file ${config_files})
  get_filename_component(basename ${file} NAME_WE)
  get_filename_component(filename ${file} NAME)
  set(target ${basename})
  kwiver_configure_file(${target}
    "${CMAKE_CURRENT_SOURCE_DIR}/${filename}"
    "${out_dir}/${filename}"
    )
  add_dependencies(kwiver-data-config configure-${target})
endforeach()

# Install config and model files
kwiver_install(
  FILES ${config_files}
  COMPONENT runtime
  DESTINATION ${kwiver_config_subdir}
  )

add_subdirectory(applets)
add_subdirectory(models)
add_subdirectory(pipelines)
