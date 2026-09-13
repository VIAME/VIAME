# Configuring a file at build time
#
# Kwiver's `kwiver_configure_file`, from `kwiver-utils-configuration.cmake`.
# One caller is left -- `version.h`, which has to be written at build time
# rather than configure time because it carries the git hash, and a configure
# that ran an hour ago would bake in the wrong one.
#
# `kwiver_symlink_file` went with the python helpers, which were its only
# user, and `kwiver_configure_file`'s other callers went with the python
# module copies: no python module VIAME installs has a `@VAR@` in it.

include_guard( GLOBAL )

if( NOT TARGET viame_configure )
  add_custom_target( viame_configure ALL )
endif()

#+
# Configure a file at build time, substituting only the named variables.
#
#   viame_configure_file( name source dest [var ...] [DEPENDS ...] )
#
# Creates a `configure-<name>` target that `viame_configure` depends on, so
# it runs as part of the default build. `viame_configure_with_git` set before
# the call exposes the repository's state to the file being configured, and
# makes the step run every time, because git state changes without any file
# this depends on changing.
#
# `__SOURCE_PATH__`, `__TEMP_PATH__` and `__OUTPUT_PATH__` are reserved.
#-
function( viame_configure_file name source dest )
  cmake_parse_arguments( CF "" "" "DEPENDS" ${ARGN} )

  set( definitions )
  foreach( arg IN LISTS CF_UNPARSED_ARGUMENTS )
    list( APPEND definitions "-D${arg}=\"${${arg}}\"" )
  endforeach()

  set( temp_file "${CMAKE_CURRENT_BINARY_DIR}/configure.${name}.output" )

  set( helper "viame-configure-helper.cmake" )
  set( stat_file )
  if( viame_configure_with_git )
    set( helper "viame-configure-git-helper.cmake" )
    # Touched every configure so the command always reruns: git state is not
    # a file this can depend on.
    set( stat_file "${CMAKE_CURRENT_BINARY_DIR}/configure.${name}.stat" )
    file( WRITE "${stat_file}"
          "Touched to force ${name} to configure." )
  endif()

  add_custom_command(
    OUTPUT  "${dest}"
    COMMAND "${CMAKE_COMMAND}"
            ${definitions}
            "-DCMAKE_MESSAGE_LOG_LEVEL:STRING=${CMAKE_MESSAGE_LOG_LEVEL}"
            "-D__SOURCE_PATH__:PATH=${source}"
            "-D__TEMP_PATH__:PATH=${temp_file}"
            "-D__OUTPUT_PATH__:PATH=${dest}"
            -P "${VIAME_CMAKE_HELPER_DIR}/${helper}"
    DEPENDS "${source}" ${CF_DEPENDS} ${stat_file}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Configuring ${name} \"${source}\" -> \"${dest}\""
    )

  set_property( DIRECTORY APPEND
    PROPERTY ADDITIONAL_MAKE_CLEAN_FILES "${temp_file}" )

  add_custom_target( configure-${name}
    DEPENDS "${dest}"
    SOURCES "${source}"
    )
  source_group( "Configured Files" FILES "${source}" )

  add_dependencies( viame_configure configure-${name} )
endfunction()
