# Configure functions for the sprokit project
# The following functions are defined:
#
#   sprokit_configure_file
#   sprokit_configure_directory
#
# Their syntax is:
#
#   sprokit_configure_file(name source dest [variable ...])
#     The first argument is the name of the file being configured. The
#     next two parameters are the source and destination paths of the file to
#     be configured. Any variables that need to be replaced in the file
#     should be passed as extra arguments. The file will be added to the
#     list of files to be cleaned.
#
#   sprokit_configure_directory(name sourcedir destdir)
#     Configures an entire directory from ${sourcedir} into ${destdir}. Add
#     a dependency on the configure-${name} to ensure that it is complete
#     before another target.

add_custom_target(configure ALL)

function (_sprokit_configure_file name source dest)
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

function (sprokit_configure_file name source dest)
  set(configure_script
    "${CMAKE_CURRENT_BINARY_DIR}/configure.${name}.cmake")
  set(configured_path
    "${configure_script}.output")

  _sprokit_configure_file(${name} "${source}" "${dest}" ${ARGN})

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
  if (NOT no_configure_target)
    add_custom_target(configure-${name} ${all}
      DEPENDS "${dest}"
      SOURCES "${source}")
    source_group("Configured Files"
      FILES "${source}")
    add_dependencies(configure
      configure-${name})
  endif ()
endfunction ()

function (sprokit_configure_file_always name source dest)
  set(extra_output
    "${dest}.noexist")

  sprokit_configure_file(${name} "${source}" "${dest}" ${ARGN})
endfunction ()

function (sprokit_configure_directory name sourcedir destdir)
  set(no_configure_target TRUE)

  file(GLOB_RECURSE sources
    RELATIVE "${sourcedir}"
    "${sourcedir}/*")

  set(count 0)
  set(source_paths)
  set(dest_paths)

  foreach (source ${sources})
    set(source_path
      "${sourcedir}/${source}")
    set(dest_path
      "${destdir}/${source}")

    sprokit_configure_file(${name}-${count}
      "${source_path}"
      "${dest_path}")

    set(source_paths
      ${source_paths}
      "${source_path}")
    set(dest_paths
      ${dest_paths}
      "${dest_path}")

    math(EXPR count "${count} + 1")
  endforeach ()

  add_custom_target(configure-${name} ${all}
    DEPENDS ${dest_paths}
    SOURCES ${source_paths})
  add_dependencies(configure
    configure-${name})
endfunction ()
