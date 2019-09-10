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
#   library_dir
#     If set, it would replace lib folder within build and install directories.
#     ( the default is 'lib' ). library_dir should not have leading or trailing
#     slashes. These slashes would be removed if they are passed in the library_dir
#
#   library_subdir
#     If set, library targets will be placed into the directory within the install
#     directory. This is necessary due to the way some systems use
#     CMAKE_BUILD_TYPE as a directory in the output path.
#
#   library_subdir_suffix
#     If set, the suffix will be appended to the subdirectory for the target.
#     This is placed after the CMAKE_BUILD_TYPE subdirectory if necessary.
#

include(CMakeParseArguments)
include (GenerateExportHeader)


# Global collection variables
define_property(GLOBAL PROPERTY kwiver_executables
  BRIEF_DOCS "KWIVER Executables"
  FULL_DOCS "List of KWIVER executables created by the kwiver_add_executable function."
  )
define_property(GLOBAL PROPERTY kwiver_executables_paths
  BRIEF_DOCS "KWIVER Executables Paths"
  FULL_DOCS "List of the binary/build paths for all KWIVER executables created by the kwiver_add_executable function."
  )
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
define_property(GLOBAL PROPERTY kwiver_plugin_path
  BRIEF_DOCS "Plugin search path"
  FULL_DOCS "List of directories to search ar run time for plugins"
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

#+
# Check if library_dir is undefined and set it to default ( lib )
#-
function(_kwiver_check_and_set_library_dir)
  if(NOT DEFINED library_dir)
    set(library_dir ${KWIVER_DEFAULT_LIBRARY_DIR} PARENT_SCOPE)
  endif()
endfunction()

#+
# Helper function to check and replace leading and trailing slashes in a path
#
# _kwiver_validate_path_value( op_path ip_path )
#
# The first argument is the path returned from the function without leading or
# trailing slashes, the second argument is the input path which may or may not
# have leading and trailing slashes
#-
function(_kwiver_validate_path_value op_path ip_path)
  if(NOT DEFINED ip_path)
    message(FATAL_ERROR, "Cannot validate undefined path ${ip_path}")
  endif()
  string(REGEX REPLACE "^/" "" ip_path "${ip_path}")
  string(REGEX REPLACE "/$" "" ip_path "${ip_path}")
  set(${op_path} "${ip_path}" PARENT_SCOPE)
endfunction()

#+
# Helper function to compute relative path of the root of a path
#
# _kwiver_path_to_root_from_lib_dir( path_to_root lib_dir )
#
# The first argument is a string that represents the relative path from a path
# to root of the path. The second argument is a path
#-
function(_kwiver_path_to_root_from_lib_dir path_to_root lib_dir)
  set(_path_to_root "")
  if(NOT DEFINED lib_dir)
    message(WARNING "Trying to determine root path for undefined variable ${lib_dir}")
  else()
    string(LENGTH "${lib_dir}" len_lib_dir)
    if(${len_lib_dir} GREATER 0)
        string(REPLACE "/" ";" library_dir_list ${lib_dir})
        list(LENGTH library_dir_list len_library_dir_list)
        if(CMAKE_VERSION VERSION_GREATER "3.15")
          string(REPEAT "../" ${len_library_dir_list} path_to_root)
        else()
          foreach(_ RANGE 1 ${len_library_dir_list})
            string(CONCAT _path_to_root "${_path_to_root}" "../")
          endforeach()
        endif()
    endif()
  endif()
  set(${path_to_root} ${_path_to_root} PARENT_SCOPE)
endfunction()

# ------------------------------
function(_kwiver_compile_pic name)
  message(STATUS "Adding PIC flag to target: ${name}")
  set_target_properties("${name}"
    PROPERTIES
      POSITION_INDEPENDENT_CODE TRUE
    )
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

  _kwiver_check_and_set_library_dir()
  _kwiver_validate_library_dir_value()

  set_target_properties(${name}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
      INSTALL_RPATH "\$ORIGIN/../${library_dir}:\$ORIGIN/"
    )

  if(NOT component)
    set(component runtime)
  endif()

  # Add to global collection variable
  set_property(GLOBAL APPEND
    PROPERTY kwiver_executables "${name}"
    )
  set_property(GLOBAL APPEND
    PROPERTY kwiver_executables_paths "${CMAKE_CURRENT_BINARY_DIR}"
    )

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

  _kwiver_check_and_set_library_dir()
  _kwiver_validate_path_value(library_dir "${library_dir}")
  _kwiver_path_to_root_from_lib_dir(lib_dir_path_to_root "${library_dir}")
  _kwiver_validate_path_value(library_subdir "${library_subdir}")
  _kwiver_path_to_root_from_lib_dir(lib_subdir_path_to_root "${library_subdir}")

  if ( APPLE )
    set( props
      MACOSX_RPATH         TRUE
      INSTALL_NAME_DIR     "@executable_path/../${library_dir}"
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
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${library_dir}${LIB_SUFFIX}${library_subdir}${library_subdir_suffix}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${library_dir}${LIB_SUFFIX}${library_subdir}${library_subdir_suffix}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin${library_subdir}${library_subdir_suffix}"
    INSTALL_RPATH            "\$ORIGIN/../${library_dir}:\$ORIGIN/"
    INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR};${CMAKE_BINARY_DIR}>$<INSTALL_INTERFACE:include>"
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
      "ARCHIVE_OUTPUT_DIRECTORY_${upper_config}" "${CMAKE_BINARY_DIR}/${library_dir}${LIB_SUFFIX}/${config}/${library_subdir}"
      "LIBRARY_OUTPUT_DIRECTORY_${upper_config}" "${CMAKE_BINARY_DIR}/${library_dir}${LIB_SUFFIX}/${config}/${library_subdir}"
      "RUNTIME_OUTPUT_DIRECTORY_${upper_config}" "${CMAKE_BINARY_DIR}/bin/${config}/${library_subdir}"
      )
  endforeach()

  if(NOT component)
    set(component runtime)
  endif()

  get_target_property(target_type "${name}" TYPE)
  if (target_type STREQUAL "STATIC_LIBRARY")
    _kwiver_compile_pic("${name}")
  endif()

  _kwiver_export("${name}")
  # LIB_SUFFIX should only apply to installation location, not the build
  # locations that properties above this point pertain to.
  kwiver_install(
    TARGETS             "${name}"
    ${exports}
    ARCHIVE DESTINATION "${library_dir}${LIB_SUFFIX}${library_subdir}"
    LIBRARY DESTINATION "${library_dir}${LIB_SUFFIX}${library_subdir}"
    RUNTIME DESTINATION "bin${library_subdir}"
    COMPONENT           ${component}
    )

  if ( NOT no_export)
    set_property(GLOBAL APPEND PROPERTY kwiver_libraries "${name}")
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
    NAMESPACE kwiver::
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
# SUBDIR - subdirectory in "lib" where plugin will be installed.
#
function( kwiver_add_plugin        name )
  set(options)
  set(oneValueArgs SUBDIR)
  set(multiValueArgs SOURCES PUBLIC PRIVATE)
  cmake_parse_arguments(PLUGIN "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

  _kwiver_check_and_set_library_dir()
  _kwiver_validate_path_value(library_dir "${library_dir}")
  _kwiver_path_to_root_from_lib_dir(lib_dir_path_to_root "${library_dir}")

  if ( PLUGIN_SUBDIR )
    set(library_subdir "/${PLUGIN_SUBDIR}") # put plugins in this subdir
  endif()

  _kwiver_validate_path_value(library_subdir "${library_subdir}")
  _kwiver_path_to_root_from_lib_dir(lib_subdir_path_to_root "${library_subdir}")


  set( no_export ON ) # do not export this product
  set( no_version ON ) # do not version plugins

  kwiver_add_library( ${name} MODULE ${PLUGIN_SOURCES} )

  target_link_libraries( ${name}
    PUBLIC        ${PLUGIN_PUBLIC}
    PRIVATE       ${PLUGIN_PRIVATE}
    )

  set_target_properties( ${name}
    PROPERTIES
      PREFIX           ""
      SUFFIX           ${CMAKE_SHARED_MODULE_SUFFIX}
      INSTALL_RPATH    "\$ORIGIN/${lib_subdir_path_to_root}${lib_dir_path_to_root}/${KWIVER_DEFAULT_LIBRARY_DIR}:\$ORIGIN/"
      )

  # Add to global collection variable
  set_property(GLOBAL APPEND
    PROPERTY kwiver_plugin_libraries    ${name}
    )

endfunction()


####
# This function adds the supplied paths to the default set of paths
# searched at **runtime** for modules.
#
# Uses the global option KWIVER_USE_CONFIGURATION_SUBDIRECTORY
# to control adding config specific directories to the path.
#
# Options are:
# SUBDIR - subdirectory in lib where plugin will be installed.
#
function( kwiver_add_module_path    dir )
    set_property(GLOBAL APPEND PROPERTY kwiver_plugin_path  "${dir}" )
endfunction()


####
# This macro creates the module directory for the plugin loader based
# on current system and other options. The resulting directory string
# is placed in the "kwiver_module_path_result" variable. Note that the
# result may be more than one path.
#
macro( kwiver_make_module_path    root subdir )
  if (WIN32)
    set(kwiver_module_path_result   "${root}/lib/${subdir}" )
    if(KWIVER_USE_CONFIGURATION_SUBDIRECTORY)
      list( APPEND  kwiver_module_path_result   "${root}/lib/$<CONFIGURATION>/${subdir}" )
    endif()
  else()  # Other Unix systems
    set(kwiver_module_path_result  "${root}/lib${LIB_SUFFIX}/${subdir}" )
  endif()
endmacro()

###
# This macro creates a symbolic link from source file to dest file.
#
add_custom_target( gen_symlinks ALL )
macro(kwiver_make_symlink src dest)
  add_custom_command(
    TARGET gen_symlinks
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${src} ${dest}
    DEPENDS  ${dest}
    COMMENT "mklink ${src} -> ${dest}")
endmacro()
