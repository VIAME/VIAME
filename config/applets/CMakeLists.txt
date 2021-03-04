set(config_files
  dump_klv.conf
  estimate_depth.conf
  fuse_depth.conf
  init_cameras_landmarks.conf
  track_features.conf
  color_mesh.conf
  )

# Ensure target directory exists
set(out_dir "${KWIVER_BINARY_DIR}/share/kwiver/${KWIVER_VERSION}/config/applets")
add_custom_command(
  OUTPUT "${out_dir}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${out_dir}"
  )

# Set up meta-target
add_custom_target(kwiver-applet-config ALL DEPENDS "${out_dir}")

# Add targets to copy config files
foreach(file ${config_files})
  get_filename_component(basename ${file} NAME_WE)
  get_filename_component(filename ${file} NAME)
  set(target "applet-${basename}")
  kwiver_configure_file(${target}
    "${CMAKE_CURRENT_SOURCE_DIR}/${filename}"
    "${out_dir}/${filename}"
    )
  add_dependencies(kwiver-applet-config configure-${target})
endforeach()

# Install config and model files
kwiver_install(
  FILES ${config_files}
  COMPONENT runtime
  DESTINATION share/kwiver/${KWIVER_VERSION}/config/applets
  )
