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
#   no_version
#       If set, the target will not have version information added to it.
#
#   component
#     If set, the target will not be installed under this component (the
#     default is 'runtime').
#
#   library_subdir
#     If set, library targets will be placed into the directory within the install
#     directory. This is necessary due to the way some systems use
#     CMAKE_BUILD_TYPE as a directory in the output path.
#
include(CMakeParseArguments)
include (GenerateExportHeader)


# Global collection variables
define_property(GLOBAL PROPERTY kwiver_export_targets
  BRIEF_DOCS "Targets exported by KWIVER"
  FULL_DOCS "List of KWIVER targets to be exported in build and install trees."
  )
define_property(GLOBAL PROPERTY kwiver_libraries
  BRIEF_DOCS "Libraries build as part of KWIVER"
  FULL_DOCS "List of static/shared libraries build by KWIVER"
  )
define_property(GLOBAL PROPERTY kwiver_plugin_libraries
  BRIEF_DOCS "Generated plugin libraries"
  FULL_DOCS "List of generated shared plugin module libraries"
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
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
      INSTALL_RPATH "\$ORIGIN/../lib:\$ORIGIN/"
    )

  if(NOT component)
    set(component runtime)
  endif()

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
# Library version will be set to that of the current PROJECT version.
#
# This function will add the library to the set of targets to be exported
# unless "no_export" was set.
#
# An export header will be created unless "no_export_header" is set.
#-
function(kwiver_add_library     name)
  string(TOUPPER "${name}" upper_name)
  message(STATUS "Making library \"${name}\"")

  add_library("${name}" ${ARGN})

  if ( APPLE )
    set( props
      MACOSX_RPATH         TRUE
      INSTALL_NAME_DIR     "@executable_path/../lib"
      )
  else()
    if ( NOT no_version ) # optional versioning
      set( props
        VERSION                  ${${CMAKE_PROJECT_NAME}_VERSION}
        SOVERSION                ${${CMAKE_PROJECT_NAME}_VERSION}
        )
    else()
      set( props )
    endif()
  endif()

  set_target_properties("${name}"
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}${library_subdir}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}${library_subdir}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin${library_subdir}"
    INSTALL_RPATH            "\$ORIGIN/../lib:\$ORIGIN/"
    ${props}
    )

  if ( NOT no_export_header )
    generate_export_header( ${name}
      STATIC_DEFINE  ${upper_name}_BUILD_AS_STATIC
      )
  endif()

  foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER "${config}" upper_config)
    set_target_properties("${name}"
      PROPERTIES
      "ARCHIVE_OUTPUT_DIRECTORY_${upper_config}" "${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}/${config}${library_subdir}"
      "LIBRARY_OUTPUT_DIRECTORY_${upper_config}" "${CMAKE_BINARY_DIR}/lib${LIB_SUFFIX}/${config}${library_subdir}"
      "RUNTIME_OUTPUT_DIRECTORY_${upper_config}" "${CMAKE_BINARY_DIR}/bin/${config}${library_subdir}"
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
  # LIB_SUFFIX should only apply to installation location, not the build
  # locations that properties above this point pertain to.
  kwiver_install(
    TARGETS             "${name}"
    ${exports}
    ARCHIVE DESTINATION lib${LIB_SUFFIX}${library_subdir}
    LIBRARY DESTINATION lib${LIB_SUFFIX}${library_subdir}
    RUNTIME DESTINATION bin${library_subdir}
    COMPONENT           ${component}
    )

  if ( NOT no_export)
    set_property(GLOBAL APPEND PROPERTY kwiver_libraries ${name})
  endif()
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
# Install KWIVER public header files to include/kwiver/...
#
# A SUBDIR may be provided in order to place the header files in a
# subdirectory under that. This path must be relative.
#
# NOPATH can be specified to ignore leading path components on the
# files being installed. This is useful when installing CMake
# generated export headers
#
# If the file name has a leading path component, it is appended to the
# install path to allow installing of headers in subdirectories.
#
#-
function(kwiver_install_headers)
  set(options NOPATH)
  set(oneValueArgs SUBDIR)
  cmake_parse_arguments(mih "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  foreach(header IN LISTS mih_UNPARSED_ARGUMENTS)
    if(mih_NOPATH)
      set( H_SUBDIR ) # use empty subdir/path to file
    else()
      get_filename_component(H_SUBDIR "${header}" DIRECTORY)
      set( H_SUBDIR "/${H_SUBDIR}" )
    endif()
    kwiver_install(
      FILES       "${header}"
      DESTINATION "include/${mih_SUBDIR}${H_SUBDIR}"
      )
  endforeach()

  # for IDE support
  source_group("Header Files\\Public"
    FILES ${mih_UNPARSED_ARGUMENTS}
    )

endfunction()


#+
# Add files to the private header source group
#
#   kwiver_private_header_group(file1 [file2 ...])
#
#-
function(kwiver_private_header_group)
  source_group("Header Files\\Private"
    FILES ${ARGN}
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


####
# This function creates a target for a loadable plugin.
#
# Options are:
# SOURCES - list of source files needed to create the plugin.
# PUBLIC - list of libraries the plugin will publically link against.
# PRIVATE - list of libraries the plugin will privately link against.
# SUBDIR - subdirectory in lib where plugin will be installed.
#
function( kwiver_add_plugin        name )
  set(options)
  set(oneValueArgs SUBDIR)
  set(multiValueArgs SOURCES PUBLIC PRIVATE)
  cmake_parse_arguments(PLUGIN "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

  if ( PLUGIN_SUBDIR )
    set(library_subdir "/${PLUGIN_SUBDIR}") # put plugins in this subdir
  endif()

  set( no_export ON ) # do not export this product

  kwiver_add_library( ${name} MODULE ${PLUGIN_SOURCES} )

  target_link_libraries( ${name}
    PUBLIC        ${PLUGIN_PUBLIC}
    PRIVATE       ${PLUGIN_PRIVATE}
    )

  set_target_properties( ${name}
    PROPERTIES
      PREFIX           ""
      SUFFIX           ${CMAKE_SHARED_MODULE_SUFFIX}
      INSTALL_RPATH    "\$ORIGIN/../../lib:\$ORIGIN/"
      )

  # Add to global collection variable
  set_property(GLOBAL APPEND
    PROPERTY kwiver_plugin_libraries    ${name}
    )

endfunction()
