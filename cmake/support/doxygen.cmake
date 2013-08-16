# Doxygen functions for the sprokit project
# The following functions are defined:
#   sprokit_create_doxygen
# Their syntax is:
#   sprokit_create_doxygen(inputdir name [tagdep...])
#     The first argument is the directory to use as the input directory for
#     doxygen. The targets `doxygen-${name}-dir', `doxygen-${name}-doxyfile',
#     `doxygen-${name}-tag', and `doxygen-${name}' will be created. All
#     `tagdep' arguments will be added as dependencies.

find_package(Doxygen)

cmake_dependent_option(SPROKIT_ENABLE_DOCUMENTATION "Build documentation" OFF
  DOXYGEN_FOUND OFF)
cmake_dependent_option(SPROKIT_INSTALL_DOCUMENTATION "Install documentation" OFF
  SPROKIT_ENABLE_DOCUMENTATION OFF)

if (DOXYGEN_FOUND)
  add_custom_target(doxygen)
endif ()

function (sprokit_create_doxygen inputdir name)
  if (SPROKIT_ENABLE_DOCUMENTATION)
    set(doxy_project_source_dir
      "${inputdir}")
    set(doxy_include_path
      "${sprokit_binary_dir}/src")
    set(doxy_documentation_output_path
      "${sprokit_binary_dir}/doc")
    set(doxy_project_name
      "${name}")
    set(doxy_tag_files)
    set(tag_targets)

    foreach (tag IN LISTS ARGN)
      list(APPEND doxy_tag_files
        "${sprokit_binary_dir}/doc/${tag}.tag=../${tag}")
      list(APPEND tag_targets
        doxygen-${tag}-tag)
    endforeach ()

    string(REPLACE ";" " " doxy_tag_files "${doxy_tag_files}")

    set(doxygen_files_dir
      "${sprokit_source_dir}/extra/doxygen")

    add_custom_target(doxygen-${name}-dir)
    add_custom_command(
      TARGET  doxygen-${name}-dir
      COMMAND "${CMAKE_COMMAND}" -E make_directory
              "${sprokit_binary_dir}/doc/${name}"
      COMMENT "Creating documentation directory for ${name}")
    sprokit_configure_file(${name}-doxyfile.common
      "${doxygen_files_dir}/Doxyfile.common.in"
      "${sprokit_binary_dir}/doc/${name}/Doxyfile.common"
      doxy_project_source_dir
      doxy_include_path
      doxy_documentation_output_path
      doxy_project_name
      doxy_tag_files
      doxy_exclude_patterns)
    sprokit_configure_file(${name}-doxyfile.tag
      "${doxygen_files_dir}/Doxyfile.tag.in"
      "${sprokit_binary_dir}/doc/${name}/Doxyfile.tag"
      doxy_documentation_output_path
      doxy_project_name)
    add_dependencies(configure-${name}-doxyfile.tag
      configure-${name}-doxyfile.common)
    sprokit_configure_file(${name}-doxyfile
      "${doxygen_files_dir}/Doxyfile.in"
      "${sprokit_binary_dir}/doc/${name}/Doxyfile"
      doxy_documentation_output_path
      doxy_project_name)
    add_dependencies(configure-${name}-doxyfile
      configure-${name}-doxyfile.common)
    add_custom_target(doxygen-${name}-tag)
    add_dependencies(doxygen-${name}-tag
      configure-${name}-doxyfile.tag)
    add_custom_command(
      TARGET  doxygen-${name}-tag
      COMMAND "${DOXYGEN_EXECUTABLE}"
              "${sprokit_binary_dir}/doc/${name}/Doxyfile.tag"
      WORKING_DIRECTORY
              "${sprokit_binary_dir}/doc/${name}"
      COMMENT "Creating tag for ${name}")
    add_custom_target(doxygen-${name})
    add_dependencies(doxygen-${name}
      configure-${name}-doxyfile
      doxygen-${name}-tag
      ${tag_targets})
    add_custom_command(
      TARGET  doxygen-${name}
      COMMAND "${DOXYGEN_EXECUTABLE}"
              "${sprokit_binary_dir}/doc/${name}/Doxyfile"
      WORKING_DIRECTORY
              "${sprokit_binary_dir}/doc/${name}"
      COMMENT "Creating documentation for ${name}")
    add_dependencies(doxygen
      doxygen-${name})

    if (SPROKIT_INSTALL_DOCUMENTATION)
      sprokit_install(
        DIRECTORY   "${sprokit_binary_dir}/doc/${name}"
        DESTINATION "share/doc/sprokit-${sprokit_version}/${name}"
        COMPONENT   documentation)
    endif ()
  endif ()
endfunction ()
