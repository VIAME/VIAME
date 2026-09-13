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
include( old_version_checks )

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
# When in super-build mode, everything is 'make installed' into these directories
# at build time ('make') instead of at ('make install') time due to the nature
# of cmake super-builds.
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
# preprocessor. Scoped to the plugin pass so fletch does not inherit the flag.
if( MSVC AND NOT VIAME_BUILD_DEPENDENCIES )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:preprocessor" )
endif()
###
# Main build loop for when we are building both VIAME and all dependencies
##
if( VIAME_BUILD_DEPENDENCIES )

  # Early error checking in case user did not run submodule init
  file( GLOB fletch_files ${VIAME_PACKAGES_DIR}/fletch/* )
  list( LENGTH fletch_files fletch_length )

  if( ${fletch_length} LESS 3 )
    message( FATAL_ERROR "Fletch directory empty, did you not run the following in \
     your source tree: \n git submodule update --init --recursive" )
  endif()

  # Include for CMake external project add command
  include( ExternalProject )

  # Specifies internal arguments passed to all projects (CXX flag passdowns, etc)
  include( common_args )

  # Core dependencies for other projects
  include( add_project_fletch )

  if( VIAME_ENABLE_PYTHON )
    include( add_project_python_deps )
  endif()

  # darknet used to be built here as an external project. It is
  # `third_party/darknet` now, compiled with the rest of the tree.

  # Kwiver is no longer an external project: P5-T05 dissolved the submodule
  # and the VIAME project below builds what was in it.
  #
  # Nor are VIVIA, SEAL, the keypoint GUI, TensorFlow or TensorRT. They were
  # five separate repositories the superbuild cloned and built beside VIAME,
  # and `lite-build-system.md` section 2 removes all five options in P1:
  # VIVIA needs VXL, which went in P3; SEAL and the keypoint GUI are GUIs
  # with their own repositories and their own build; TensorFlow and TensorRT
  # are inference backends nothing in the shipped pipelines selects.

  if( VIAME_ENABLE_DIVE )
    include( add_project_dive )
  endif()

  if ( VIAME_ENABLE_ONNX)
    include( add_project_onnx )
  endif()

  if( VIAME_ENABLE_PYTORCH )
    include( add_project_pytorch )
  endif()

  # VIAME itself, which re-invokes CMake on this source tree with
  # `VIAME_BUILD_DEPENDENCIES` off, so that `cmake/viame_project.cmake` runs
  include( add_project_viame )

  # Fixup installer package
  if( VIAME_FIXUP_BUNDLE )
    include( build_package )
  endif()

  # Register tests at super-build level
  # Note: The actual CTestTestfile.cmake is created by the setup_ctest step
  # in add_project_viame.cmake after viame-build completes
  if( VIAME_ENABLE_TESTS )
    enable_testing()
    include( CTest )
  endif()
else()

  include( viame_project )

endif()
