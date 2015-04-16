#
# KWIVER Target creation and installation support
#
# Variables that affect behavior of functions:
#
#   no_export
#       if set, target will not be exported.
#
#   no_install
#       If set, target will not be installed.
#
#   component
#     If set, the target will not be installed under this component (the
#     default is 'runtime').
#
#   library_subdir
#     If set, library targets will be placed into the directory with this
#     as a suffix. This is necessary due to the way some systems use
#     CMAKE_BUILD_TYPE as a directory in the output path.
#
include(CMakeParseArguments)

# Global collection variables
define_property(GLOBAL PROPERTY kwiver_export_targets
  BRIEF_DOCS "Targets exported by KWIVER"
  FULL_DOCS "List of KWIVER targets to be exported in build and install trees."
  )
define_property(GLOBAL PROPERTY kwiver_libraries
  BRIEF_DOCS "Libraries build as part of KWIVER"
  FULL_DOCS "List of static/shared libraries build by KWIVER"
  )


#+
# Helper function to manage export string string generation and the no_export
# flag.
#
# Sets the variable "exports" which should be expanded into the install
# command.
#-
function(_kwiver_export name)
  set(exports)
  if(no_export)
    return()
  endif()
  set(exports
    EXPORT ${kwiver_export_name}
    PARENT_SCOPE
    )
  set_property(GLOBAL APPEND PROPERTY kwiver_export_targets ${name})
endfunction()


# ------------------------------
function(_kwiver_compile_pic name)
  message(STATUS "Adding PIC flag to target: ${name}")
  if (CMAKE_VERSION VERSION_GREATER "2.8.12")
    set_target_properties("${name}"
      PROPERTIES
        POSITION_INDEPENDENT_CODE TRUE
      )
  elseif(NOT MSVC)
    set_target_properties("${name}"
      PROPERTIES
        COMPILE_FLAGS "-fPIC"
      )
  endif()
endfunction()


#+
# Wrapper around install(...) that catches ``no_install`` if set
#
#   kwiver_install([args])
#
# All args given to this function are passed directly to install(...),
# provided ``no_install`` is not set. See CMake documentation for
# install(...) usage.
#-
function(kwiver_install)
  if(no_install)
    return()
  endif()

  install(${ARGN})
endfunction()


#+
# Add an executable to KWIVER
#
#   kwiver_add_executable(name [args...])
#
# All args given to this function are passed to CMake's add_executable(...)
# function after providing the name, so refer to CMake's documentation for
# additional valid arguments.
#
# This function will add the executable to the set of targets to be exported
# unless ``no_export`` was set.
#-
function(kwiver_add_executable name)
  add_executable(${name} ${ARGN})
  set_target_properties(${name}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${KWIVER_BINARY_DIR}/bin"
    )

  if(NOT component)
    set(component runtime)
  endif()

  _kwiver_export(${name})
  kwiver_install(
    TARGETS     ${name}
    ${exports}
    DESTINATION bin
    COMPONENT   ${component}
    )
endfunction()


#+
# Add a library to KWIVER
#
#   kwiver_add_library(name [args...])
#
# Remaining arguments passed to this function are given to the underlying
# add_library call, so refer to CMake documentation for additional arguments.
#
# Library version will be set to that of the current KWIVER version.
# Additionally defines the symbol "MAKE_<cname>_LIB" where ``cname`` is the
# ``name`` capitalized.
#
# This function will add the library to the set of targets to be exported
# unless ``no_export`` was set.
#-
function(kwiver_add_library name)
  string(TOUPPER "${name}" upper_name)
  message(STATUS "Making library \"${name}\" with defined symbol \"MAKE_${upper_name}_LIB\"")

  add_library("${name}" ${ARGN})
  set_target_properties("${name}"
    PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "${KWIVER_BINARY_DIR}/lib${library_subdir}"
      LIBRARY_OUTPUT_DIRECTORY "${KWIVER_BINARY_DIR}/lib${library_subdir}"
      RUNTIME_OUTPUT_DIRECTORY "${KWIVER_BINARY_DIR}/bin${library_subdir}"
      VERSION                  ${KWIVER_VERSION}
      SOVERSION                0
      DEFINE_SYMBOL            MAKE_${upper_name}_LIB
    )

  add_dependencies( "${name}"
    configure-exim_config.h
#??    configure-modules.h
    )

  foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER "${config}" upper_config)
    set_target_properties("${name}"
      PROPERTIES
        "ARCHIVE_OUTPUT_DIRECTORY_${upper_config}" "${KWIVER_BINARY_DIR}/lib/${config}${library_subdir}"
        "LIBRARY_OUTPUT_DIRECTORY_${upper_config}" "${KWIVER_BINARY_DIR}/lib/${config}${library_subdir}"
        "RUNTIME_OUTPUT_DIRECTORY_${upper_config}" "${KWIVER_BINARY_DIR}/bin/${config}${library_subdir}"
      )
  endforeach()

  if(NOT component)
    set(component runtime)
  endif()

  get_target_property(target_type "${name}" TYPE)
  if (target_type STREQUAL "STATIC_LIBRARY")
    _kwiver_compile_pic("${name}")
  endif()

  _kwiver_export(${name})
  # MATPK_LIB_SUFFIX should only apply to installation location, not the build
  # locations that properties above this point pertain to.
  kwiver_install(
    TARGETS             "${name}"
    ${exports}
    ARCHIVE DESTINATION lib${KWIVER_LIB_SUFFIX}${library_subdir}
    LIBRARY DESTINATION lib${KWIVER_LIB_SUFFIX}${library_subdir}
    RUNTIME DESTINATION bin${library_subdir}
    COMPONENT           ${component}
    )

  set_property(GLOBAL APPEND PROPERTY kwiver_libraries ${name})
endfunction()


#+
#   kwiver_export_targets(file [APPEND])
#
# Export all recorded KWIVER targets to the given file in the build tree. If
# there are no targets recorded, this is a no-op. APPEND may be give to tell
# us to append to the given file instead of overwriting it.
#-
function(kwiver_export_targets file)
  get_property(export_targets GLOBAL PROPERTY kwiver_export_targets)
  export(
    TARGETS ${export_targets}
    ${ARGN}
    FILE "${file}"
    )
  #message(STATUS "Adding to file to clean: ${file}")
  #set_directory_properties(
  #  PROPERTIES
  #    ADDITIONAL_MAKE_CLEAN_FILES "${file}"
  #  )
endfunction()


#+
#   kwiver_install_headers(header1 [header2 ...] [SUBDIR dir])
#
# Install KWIVER public header files to include/kwiver.
#
# A SUBDIR may be provided in order to place the header files in a
# subdirectory under that. This path must be relative.
#-
function(kwiver_install_headers)
  set(oneValueArgs SUBDIR)
  cmake_parse_arguments(mih "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  #kwiver_install(
  #  FILES       ${mih_UNPARSED_ARGUMENTS}
  #  DESTINATION "include/kwiver/${mih_SUBDIR}"
  #  )
  foreach(header IN LISTS mih_UNPARSED_ARGUMENTS)
    get_filename_component(H_SUBDIR "${header}" PATH)
    kwiver_install(
      FILES       "${header}"
      DESTINATION "include/kwiver/${mih_SUBDIR}/${H_SUBDIR}"
      )
  endforeach()
endfunction()


#+
# Add files to the private header source group
#
#   kwiver_private_header_group(file1 [file2 ...])
#
#-
function(kwiver_private_header_group)
  source_group("Header Files\\Private"
    ${ARGN}
    )
endfunction()


#+
# Add files to the private template group
#
#   kwiver_private_template_group(file1 [file2 ...])
#
#-
function(kwiver_private_template_group)
  source_group("Template Files\\Private"
    ${ARGN}
    )
endfunction()
