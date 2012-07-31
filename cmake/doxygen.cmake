# Doxygen functions for the vistk project
# The following functions are defined:
#   create_doxygen
# Their syntax is:
#   create_doxygen(inputdir name [tagdep...])
#     The first argument is the directory to use as the input directory for
#     doxygen. The targets `doxygen-${name}-dir', `doxygen-${name}-doxyfile',
#     `doxygen-${name}-tag', and `doxygen-${name}' will be created. All
#     `tagdep' arguments will be added as dependencies.

find_package(Doxygen)

if (DOXYGEN_FOUND)
  option(VISTK_ENABLE_DOCUMENTATION "Build documentation" OFF)

  add_custom_target(doxygen)

  if (VISTK_ENABLE_DOCUMENTATION)
    option(VISTK_INSTALL_DOCUMENTATION "Install documentation" OFF)
  endif ()
endif ()

function (create_doxygen inputdir name)
  if (VISTK_ENABLE_DOCUMENTATION)
    set(doxy_project_source_dir
      "${inputdir}")
    set(doxy_documentation_output_path
      "${vistk_binary_dir}/doc")
    set(doxy_project_name
      "${name}")
    set(doxy_tag_files)
    set(tag_targets)

    foreach (tag ${ARGN})
      set(doxy_tag_files
        ${doxy_tag_files} "${vistk_binary_dir}/doc/${tag}.tag=../${tag}")
      set(tag_targets
        ${tag_targets}
        doxygen-${tag}-tag)
    endforeach ()

    string(REPLACE ";" " " doxy_tag_files "${doxy_tag_files}")

    add_custom_target(doxygen-${name}-dir)
    add_custom_command(
      TARGET  doxygen-${name}-dir
      COMMAND "${CMAKE_COMMAND}" -E make_directory
              "${vistk_binary_dir}/doc/${name}"
      COMMENT "Creating documentation directory for ${name}")
    vistk_configure_file(${name}-doxyfile.common
      "${vistk_source_dir}/cmake/Doxyfile.common.in"
      "${vistk_binary_dir}/doc/${name}/Doxyfile.common"
      doxy_project_source_dir
      doxy_documentation_output_path
      doxy_project_name
      doxy_tag_files
      doxy_exclude_patterns)
    vistk_configure_file(${name}-doxyfile.tag
      "${vistk_source_dir}/cmake/Doxyfile.tag.in"
      "${vistk_binary_dir}/doc/${name}/Doxyfile.tag"
      doxy_documentation_output_path
      doxy_project_name)
    add_dependencies(configure-${name}-doxyfile.tag
      configure-${name}-doxyfile.common)
    vistk_configure_file(${name}-doxyfile
      "${vistk_source_dir}/cmake/Doxyfile.in"
      "${vistk_binary_dir}/doc/${name}/Doxyfile"
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
              "${vistk_binary_dir}/doc/${name}/Doxyfile.tag"
      WORKING_DIRECTORY
              "${vistk_binary_dir}/doc/${name}"
      COMMENT "Creating tag for ${name}")
    add_custom_target(doxygen-${name})
    add_dependencies(doxygen-${name}
      configure-${name}-doxyfile
      doxygen-${name}-tag
      ${tag_targets})
    add_custom_command(
      TARGET  doxygen-${name}
      COMMAND "${DOXYGEN_EXECUTABLE}"
              "${vistk_binary_dir}/doc/${name}/Doxyfile"
      WORKING_DIRECTORY
              "${vistk_binary_dir}/doc/${name}"
      COMMENT "Creating documentation for ${name}")
    add_dependencies(doxygen
      doxygen-${name})

    if (VISTK_INSTALL_DOCUMENTATION)
      vistk_install(
        DIRECTORY   "${vistk_binary_dir}/doc/${name}"
        DESTINATION "share/doc/vistk-${vistk_version}/${name}"
        COMPONENT   documentation)
    endif ()
  endif ()
endfunction ()
