# Doxygen functions for the vistk project
# The following functions are defined:
#   create_doxygen
# Their syntax is:
#   create_doxygen(inputdir name [tagdep...])
#     The first argument is the directory to use as the input directory for
#     doxygen. The targets `doxygen-${name}-dir', `doxygen-${name}-doxyfile',
#     `doxygen-${name}-tag', and `doxygen-${name}' will be created. All
#     `tagdep' arguments will be added as dependencies.

find_package(Doxygen REQUIRED)

add_custom_target(doxygen)
option(VISTK_INSTALL_DOCUMENTATION "Install documentation" OFF)

function (create_doxygen inputdir name)
  set(tag_files)
  set(tag_targets)

  foreach (tag ${ARGN})
    set(tag_files
      ${tag_files} ${DOCUMENTATION_OUTPUT_PATH}/${tag}.tag=../${tag})
    set(tag_targets
      ${tag_targets}
      doxygen-${tag}-tag)
  endforeach (tag)

  string(REPLACE ";" " " tag_files "${tag_files}")

  add_custom_target(doxygen-${name}-dir)
  add_custom_command(
    TARGET  doxygen-${name}-dir
    COMMAND cmake -E make_directory ${DOCUMENTATION_OUTPUT_PATH}/${name}
    COMMENT "Creating documentation directory for ${name}")
  add_custom_target(doxygen-${name}-doxyfile)
  add_dependencies(doxygen-${name}-doxyfile
    doxygen-${name}-dir)
  add_custom_command(
    TARGET  doxygen-${name}-doxyfile
    COMMAND ${CMAKE_COMMAND}
            -D "DOXYGEN_TEMPLATE=${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in"
            -D "DOXY_PROJECT_SOURCE_DIR=${inputdir}"
            -D "DOXY_DOCUMENTATION_OUTPUT_PATH=${DOCUMENTATION_OUTPUT_PATH}"
            -D "DOXY_PROJECT_NAME=${name}"
            -D "DOXY_TAG_FILES=\"${tag_files}\""
            -P "${CMAKE_SOURCE_DIR}/cmake/doxygen-script.cmake"
    WORKING_DIRECTORY
            ${DOCUMENTATION_OUTPUT_PATH}/${name}
    COMMENT "Generating Doxyfile for ${name}")
  add_custom_target(doxygen-${name}-tag)
  add_dependencies(doxygen-${name}-tag
    doxygen-${name}-doxyfile)
  add_custom_command(
    TARGET  doxygen-${name}-tag
    COMMAND ${DOXYGEN_EXECUTABLE}
    WORKING_DIRECTORY
            ${DOCUMENTATION_OUTPUT_PATH}/${name}
    COMMENT "Creating tag for ${name}")
  add_custom_target(doxygen-${name})
  add_dependencies(doxygen-${name}
    doxygen-${name}-tag
    ${tag_targets})
  add_custom_command(
    TARGET  doxygen-${name}
    COMMAND ${DOXYGEN_EXECUTABLE}
    WORKING_DIRECTORY
            ${DOCUMENTATION_OUTPUT_PATH}/${name}
    COMMENT "Creating documentation for ${name}")
  add_dependencies(doxygen
    doxygen-${name})

  if (VISTK_INSTALL_DOCUMENTATION)
    install(
      DIRECTORY   ${DOCUMENTATION_OUTPUT_PATH}/${name}
      DESTINATION ${DOCUMENTATION_INSTALL_PATH}/${CMAKE_PROJECT_NAME}-${vistk_version}/${name}
      COMPONENT   documentation)
  endif (VISTK_INSTALL_DOCUMENTATION)
endfunction (create_doxygen)
