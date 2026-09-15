# What kwiver's top-level CMakeLists still did for the imported code
#
# P5-T05 dissolved `packages/kwiver`. Everything that directory's CMakeLists
# set up which the imported vital, sprokit and python package still need is
# here, in VIAME's own scope. What is not here is what kwiver did for itself:
# its plugin subdirectories, its export name and its config package (VIAME
# has its own), its CPack setup, and the dependency lookups for arrows that
# no longer exist.

###
# Version
#
# The install lays out `share/kwiver/<version>/config`, and the pipeline and
# cluster search paths are written into the binaries from it, so the version
# is kwiver's until phase 11 renames the tree.
if( NOT KWIVER_VERSION )
  file( READ "${CMAKE_CURRENT_LIST_DIR}/VERSION.txt" KWIVER_VERSION )
  string( STRIP "${KWIVER_VERSION}" KWIVER_VERSION )
endif()

string( REGEX MATCH "^([0-9]+)\.([0-9]+)\.([0-9]+)" _ "${KWIVER_VERSION}" )
set( KWIVER_VERSION_MAJOR "${CMAKE_MATCH_1}" )
set( KWIVER_VERSION_MINOR "${CMAKE_MATCH_2}" )
set( KWIVER_VERSION_PATCH "${CMAKE_MATCH_3}" )

set( KWIVER_DEFAULT_LIBRARY_DIR "lib" CACHE STRING
     "Default library directory for kwiver" )
mark_as_advanced( KWIVER_DEFAULT_LIBRARY_DIR )

set( LIB_SUFFIX "" CACHE STRING
  "Library directory suffix. e.g. suffix=\"kwiver\" will install libraries in \"libkwiver\" rather than \"lib\"" )
mark_as_advanced( LIB_SUFFIX )

set( kwiver_config_subdir share/kwiver/${KWIVER_VERSION}/config )

add_definitions( -DKWIVER_DEFAULT_PLUGIN_ORGANIZATION="Kitware Inc." )


###
# vital/version.h
#
# `viame_configure_file` writes it with the `viame_configure` target at
# build time, which is why the second one is configured rather than copied:
# at configure time there is nothing to copy.
set( viame_configure_with_git on )

viame_configure_file( version.h
  "${VIAME_LITE_LIBRARY_DIR}/algorithm_framework/version.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/vital/version.h"
  KWIVER_VERSION_MAJOR
  KWIVER_VERSION_MINOR
  KWIVER_VERSION_PATCH
  KWIVER_VERSION
  KWIVER_SOURCE_DIR
  )

viame_configure_file( viame_version.h
  "${VIAME_LITE_LIBRARY_DIR}/algorithm_framework/version.h.in"
  "${VIAME_LITE_GENERATED_DIR}/viame/algorithm_framework/version.h"
  KWIVER_VERSION_MAJOR
  KWIVER_VERSION_MINOR
  KWIVER_VERSION_PATCH
  KWIVER_VERSION
  KWIVER_SOURCE_DIR
  )

viame_install_headers(
  "${VIAME_LITE_GENERATED_DIR}/viame/algorithm_framework/version.h"
  SUBDIR viame/algorithm_framework
  NOPATH )

set( kwiver_configure_with_git )

###
# The plugin search path and the pipe and cluster search paths, compiled in
##
if( WIN32 )
  set( path_sep "\\073" )
else()
  set( path_sep "\\072" )
endif()

set( KWIVER_DEFAULT_MODULE_PATHS ""
  CACHE STRING "The default paths for module scanning. Separate paths with ';' character." FORCE )
mark_as_advanced( KWIVER_DEFAULT_MODULE_PATHS )

set( SPROKIT_DEFAULT_PIPE_INCLUDE_PATHS
  "${CMAKE_INSTALL_PREFIX}/${kwiver_config_subdir}/pipelines/include"
  CACHE STRING "The default paths to search for pipe includes in" FORCE )

configure_file(
  "${VIAME_LITE_LIBRARY_DIR}/algorithm_framework/kwiver-include-paths.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/vital/kwiver-include-paths.h" )

# The imported plugin loader includes it under the new prefix
configure_file(
  "${VIAME_LITE_LIBRARY_DIR}/algorithm_framework/kwiver-include-paths.h.in"
  "${VIAME_LITE_GENERATED_DIR}/viame/algorithm_framework/kwiver-include-paths.h" )
