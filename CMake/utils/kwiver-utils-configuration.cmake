#
# Helper functions for CMake configuring files
#
# Variables that affect behavior:
#
#   no_configure_target
#       If defined, configuration actions do not add explicit build targets.
#
include(CMakeParseArguments)

# Top level configuration target
add_custom_target(kwiver_configure ALL)

#+
# Configure the given sourcefile to the given destfile
#
#   kwiver_configure_file( name sourcefile destfile [var1 [var2 ...]]
#                         [DEPENDS ...] )
#
# Configure a sourcefile with a given name into the given destfile. Only the
# given variables (var1, var2, etc.) will be considered for replacement during
# configuration.
#
# This functions by generating custom configuration files for each call that
# controls the configuration. Generated files are marked for cleaning.
#
# If "no_configure_target" is NOT set, this creates a target of the form
# "configure-<name>" for this configuration step.
#
# Additional configuration dependencies may be set with the DEPENDS and are
# passed to the underlying ``add_custom_command``.
#
# The special symbols ``__OUTPUT_PATH__``, ``__TEMP_PATH__``, and
# ``__SOURCE_PATH__`` are reserved by this method for additional configuration
# purposes, so don't use them as configuration variables in the file you are
# trying to configure.
#-
function(kwiver_configure_file name source dest)
  set(multiValueArgs DEPENDS)
  cmake_parse_arguments(mcf "" "" "${multiValueArgs}" ${ARGN})

  message(STATUS "[configure-${name}] Creating configure command")

  set(gen_command_args)
  foreach(arg IN LISTS mcf_UNPARSED_ARGUMENTS)
    list(APPEND gen_command_args
      "-D${arg}=\"${${arg}}\""
      )
  endforeach()
  set(temp_file "${CMAKE_CURRENT_BINARY_DIR}/configure.${name}.output")

  set(KWIVER_CONFIG_HELPER "kwiver-configure-helper.cmake")
  if(kwiver_configure_with_git)
    set(KWIVER_CONFIG_HELPER "kwiver-configure-git-helper.cmake")
    # touch this status file to force the configuration to always run
    # this is needed so that Git will run and detect repository state
    # changes
    set(stat_file "${CMAKE_CURRENT_BINARY_DIR}/configure.${name}.stat")
    file(WRITE "${stat_file}"
         "This file is touched to force ${name} to configure.")
  endif()

  add_custom_command(
    OUTPUT  "${dest}"
    COMMAND "${CMAKE_COMMAND}"
            ${gen_command_args}
            "-D__SOURCE_PATH__:PATH=${source}"
            "-D__TEMP_PATH__:PATH=${temp_file}"
            "-D__OUTPUT_PATH__:PATH=${dest}"
            -P "${KWIVER_CMAKE_ROOT}/tools/${KWIVER_CONFIG_HELPER}"
    DEPENDS
            "${source}" ${mcf_DEPENDS} ${stat_file}
    WORKING_DIRECTORY
            "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Configuring ${name} file \"${source}\" -> \"${dest}\""
    )
  # also clean the intermediate generated file
  set_property(DIRECTORY APPEND PROPERTY
    ADDITIONAL_MAKE_CLEAN_FILES "${temp_file}"
    )

  # This passes if not defined or a false-evaluating value
  if(NOT no_configure_target)
    add_custom_target(configure-${name}
      DEPENDS "${dest}"
      SOURCES "${source}"   # Addding source for IDE purposes
      )

    source_group("Configured Files"
      FILES "${source}"
      )

    add_dependencies(kwiver_configure
      configure-${name}
      )
  endif()
endfunction()


###
#
# Mimics a `kwiver_configure_file`, but will symlink `source` to `dest`
# directly without any configureation. This should only be used for interpreted
# languages like python to prevent the need to re-make the project after making
# small changes to these interpreted files.
#
# SeeAlso:
#     kwiver/CMake/utils/kwiver-utils-python.cmake
#     kwiver/sprokit/conf/sprokit-macro-configure.cmake
#
function (kwiver_symlink_file name source dest)

  if(EXISTS ${dest} AND NOT IS_SYMLINK ${dest})
    # If our target it not a symlink, then remove it so we can replace it
    file(REMOVE ${dest})
  endif()

  # Need to ensure the directory exists before we create a symlink there
  get_filename_component(dest_dir ${dest} DIRECTORY)
  add_custom_command(
    OUTPUT  "${dest_dir}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory ${dest_dir}
    )

  add_custom_command(
    OUTPUT  "${dest}"
    COMMAND "${CMAKE_COMMAND}" -E create_symlink ${source} ${dest}
    DEPENDS "${source}" "${dest_dir}"
    COMMENT "Symlink-configuring ${name} file \"${source}\" -> \"${dest}\""
    )


  # This passes if not defined or a false-evaluating value
  if(NOT no_configure_target)
    add_custom_target(configure-${name}
      DEPENDS "${dest}"
      SOURCES "${source}"   # Adding source for IDE purposes
      )

    source_group("Configured Files"
      FILES "${source}"
      )

    add_dependencies(kwiver_configure
      configure-${name}
      )
  endif()
endfunction ()
