# Configure functions for the vistk project
# The following functions are defined:
#   vistk_configure_file
#   vistk_configure_pkgconfig
# Their syntax is:
#   vistk_configure_file(name source dest [variable ...])
#     The first argument is the name of the file being configured. The next two
#     parameters are the source and destination paths of the file to be
#     configured. Any variables that need to be replaced in the file should be
#     passed as extra arguments. The file will be added to the list of files to
#     be cleaned.
#   vistk_configure_pkgconfig(module)
#     A convenience function for creating pkgconfig files.

add_custom_target(configure ALL)

function (_vistk_configure_file name source dest)
  file(WRITE "${configure_script}"
    "# Configure script for \"${source}\" -> \"${dest}\"\n")

  foreach (arg ${ARGN})
    file(APPEND "${configure_script}"
      "set(${arg} \"${${arg}}\")\n")
  endforeach ()

  file(APPEND "${configure_script}" "${configure_code}")

  file(APPEND "${configure_script}" "
configure_file(
  \"${source}\"
  \"${configured_path}\"
  @ONLY)\n")

  file(APPEND "${configure_script}" "
configure_file(
  \"${configured_path}\"
  \"${dest}\"
  COPYONLY)\n")

  set(clean_files
    "${dest}"
    "${configured_path}"
    "${configure_script}")

  set_directory_properties(
    PROPERTIES
      ADDITIONAL_MAKE_CLEAN_FILES "${clean_files}")
endfunction ()

function (vistk_configure_file name source dest)
  set(configure_script
    "${CMAKE_CURRENT_BINARY_DIR}/configure.${name}.cmake")
  set(configured_path
    "${configure_script}.output")

  _vistk_configure_file(${name} "${source}" "${dest}" ${ARGN})

  add_custom_command(
    OUTPUT  "${dest}"
            ${extra_output}
    COMMAND "${CMAKE_COMMAND}"
            -P "${configure_script}"
    MAIN_DEPENDENCY
            "${source}"
    DEPENDS "${source}"
            "${configure_script}"
    WORKING_DIRECTORY
            "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Configuring ${name} file \"${source}\" -> \"${dest}\"")
  add_custom_target(configure-${name} ${all}
    DEPENDS "${dest}"
    SOURCES "${source}")
  source_group("Configured Files"
    FILES "${source}")
  add_dependencies(configure
    configure-${name})
endfunction ()

function (vistk_configure_file_always name source dest)
  set(extra_output
    "${dest}.noexist")

  vistk_configure_file(${name} "${source}" "${dest}" ${ARGN})
endfunction ()

function (vistk_configure_pkgconfig module)
  if (UNIX)
    set(pkgconfig_file "${vistk_binary_dir}/${module}.pc")

    vistk_configure_file(vistk-${module}.pc
      "${CMAKE_CURRENT_SOURCE_DIR}/${module}.pc.in"
      "${pkgconfig_file}"
      vistk_version
      CMAKE_INSTALL_PREFIX
      LIB_SUFFIX
      ${ARGN})

    install(
      FILES       "${pkgconfig_file}"
      DESTINATION "lib${LIB_SUFFIX}/pkgconfig"
      COMPONENT   development)
  endif ()
endfunction ()
