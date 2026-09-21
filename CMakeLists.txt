###    (V) (°,,,°) (V)     <><     (V) (°,,,°) (V)    <><     (V) (°,,,°) (V)
# Top-level CMakeLists.txt build file for the VIAME super-build and project
##  ) (V)    <><     (V) (°,,,°) (V)     <><     (V) (°,,,°) (V)    <><

cmake_minimum_required( VERSION 3.16.0 )
set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )

project( VIAME )

###
# Core versioning flags
##

set( VIAME_VERSION_MAJOR 1 )
set( VIAME_VERSION_MINOR 0 )
set( VIAME_VERSION_PATCH 0 )

set( VIAME_VERSION "${VIAME_VERSION_MAJOR}.${VIAME_VERSION_MINOR}.${VIAME_VERSION_PATCH}" )

###
# Add global cmake properties
##
include( CMakeDependentOption )

if( POLICY CMP0053 )
  cmake_policy( SET CMP0053 NEW )
endif()

if( POLICY CMP0074 )
  cmake_policy( SET CMP0074 NEW )
endif()

if( POLICY CMP0144 )
  cmake_policy( SET CMP0144 NEW )
endif()

if( POLICY CMP0135 )
  cmake_policy( SET CMP0135 NEW )
endif()

if( POLICY CMP0146 )
  cmake_policy( SET CMP0146 OLD )
endif()

###
# Add script folders for cmake functions
##
# `KWIVER_SOURCE_DIR` and `KWIVER_BINARY_DIR` are still read by the imported
# sources' own CMakeLists and by `viame-setup-python`, which derives the
# python output path from the binary one, so they are set before anything
# that needs them. They used to be the submodule's directories; they are
# VIAME's since P5-T05, and phase 11 renames them.
set( KWIVER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" )
set( KWIVER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" )

list( INSERT CMAKE_MODULE_PATH 0
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/viame" )

###
# Helper macros used across all cmakelists files in this project
##
include( common_macros )
include( viame_lite_import )
include( viame_static_registration )

###
# Add other common paths usable throughout all CMakeLists files
##
set( VIAME_SOURCE_DIR               "${CMAKE_SOURCE_DIR}" )
set( VIAME_CMAKE_DIR                "${CMAKE_SOURCE_DIR}/cmake" )
set( VIAME_BINARY_DIR               "${CMAKE_BINARY_DIR}/build" )
set( VIAME_BUILD_PREFIX             "${CMAKE_BINARY_DIR}/build" )
set( VIAME_PACKAGES_DIR             "${VIAME_SOURCE_DIR}/packages" )
set( VIAME_PATCHES_DIR              "${VIAME_PACKAGES_DIR}/patches" )
set( VIAME_DOWNLOAD_DIR             "${VIAME_PACKAGES_DIR}/downloads" )

###
# Add paths which control the output directories of all final install files
#
# The libraries, configs and setup scripts get here with `make install` (or
# `cmake --install`), not with `make`. Configure and build do write some things
# here themselves: downloaded model packs, the standalone python, DIVE and the
# python packages.
##
set( VIAME_BUILD_INSTALL_PREFIX     "${CMAKE_BINARY_DIR}/install"
     CACHE STRING                   "Installation directory to place all final output files" )
mark_as_advanced( VIAME_BUILD_INSTALL_PREFIX )

set( VIAME_INSTALL_PREFIX           "${VIAME_BUILD_INSTALL_PREFIX}"
     CACHE INTERNAL                 "Synonym for VIAME_BUILD_INSTALL_PREFIX" FORCE )
set( CMAKE_INSTALL_PREFIX           "${VIAME_BUILD_INSTALL_PREFIX}"
     CACHE INTERNAL                 "Synonym for VIAME_BUILD_INSTALL_PREFIX" FORCE )

###
# VIAME's options
##
include( viame_options )

###
# What the build needs to find
##
include( viame_dependencies )

# Set default windows configuration mode to be 'Release'
set( VIAME_FIRST_CONFIGURATION TRUE CACHE BOOL "Has this project been configured." )
mark_as_advanced( VIAME_FIRST_CONFIGURATION )

if( MSVC AND VIAME_FIRST_CONFIGURATION )
  set( CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "This specifies what \
    build types (configurations) will be available for selection within \
	  Visual Studio. Options are: Release, RelWithDebInfo, Debug etc. in a \
	  semi-colon seperated list." FORCE )

  set( VIAME_FIRST_CONFIGURATION FALSE CACHE BOOL "Project has been configured." FORCE )
endif()

if( MSVC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "18.00" )
  message( FATAL_ERROR "You must use at least MSVC version 13 or greater" )
endif()

# KWIVER's pluggable macros expand to nothing under the traditional MSVC
# preprocessor. This was scoped to the plugin pass so that fletch did not
# inherit the flag; there is no fletch and no second pass.
if( MSVC )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:preprocessor" )
endif()

###
# The VIAME project build
##
# `if( VIAME_BUILD_DEPENDENCIES )` stood here, and above it four hundred
# lines that built VIAME's dependencies as ExternalProjects and then
# re-invoked CMake on this same source tree with the option off, so that
# the `else()` branch -- the actual VIAME build -- ran as a second,
# separate configure.
#
# There are no dependencies left to build. Fletch went when nothing linked
# it, the five GUI and inference-backend projects went with their options in
# P1-T02, kwiver became `library/` in P5-T05, darknet and the rest became
# `library/tpl/`, the python environment is lock files, and DIVE builds from
# its submodule in this configure rather than a nested one.
##

include( viame_project )


###
# Wheel packaging
#
# Last, because it packs the install prefix and therefore wants every
# `install()` rule above it to have been declared. The target is not in
# `all`: see cmake/wheel/wheel.cmake.
##

include( ${VIAME_CMAKE_DIR}/wheel/wheel.cmake )
