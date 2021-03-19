set(config_files
  default_burnout_600_iters
  )

# Ensure target directory exists
set(out_dir "${KWIVER_BINARY_DIR}/share/kwiver/${KWIVER_VERSION}/config/models")
add_custom_command(
  OUTPUT "${out_dir}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${out_dir}"
  )

# Set up meta-target
add_custom_target(kwiver-models ALL DEPENDS "${out_dir}")

# Add targets to copy config files
foreach(file ${config_files})
  get_filename_component(basename ${file} NAME_WE)
  get_filename_component(filename ${file} NAME)
  set(target "models-${basename}")
  kwiver_configure_file(${target}
    "${CMAKE_CURRENT_SOURCE_DIR}/${filename}"
    "${out_dir}/${filename}"
    )
  add_dependencies(kwiver-models configure-${target})
endforeach()

# Install config and model files
kwiver_install(
  FILES ${config_files}
  COMPONENT runtime
  DESTINATION share/kwiver/${KWIVER_VERSION}/config/models
  )
