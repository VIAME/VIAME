###
# top level build file for NOAA VIAME
#

cmake_minimum_required( VERSION 3.0 )

project( VIAME )

include( CMakeDependentOption )

###
# Add paths used throughout CMakeLists files
##
list( INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/CMake" "${KWIVER_CMAKE_DIR}" )

set( VIAME_SOURCE_DIR            "${CMAKE_SOURCE_DIR}" )
set( VIAME_BINARY_DIR            "${CMAKE_BINARY_DIR}/build" )
set( VIAME_BUILD_PREFIX          "${CMAKE_BINARY_DIR}/build" )
set( VIAME_BUILD_INSTALL_PREFIX  "${CMAKE_BINARY_DIR}/install" )

set( VIAME_PACKAGES_DIR          "${VIAME_SOURCE_DIR}/packages" )

###
# Add plugin enable flags
##
option( VIAME_ENABLE_VXL         "Enable VXL plugins"       OFF )
option( VIAME_ENABLE_CAFFE       "Enable Caffe plugins"     OFF )
option( VIAME_ENABLE_OPENCV      "Enable OpenCV plugins"    OFF )
option( VIAME_ENABLE_MATLAB      "Enable Matlab plugins"    OFF )
option( VIAME_ENABLE_PYTHON      "Enable Python plugins"    OFF )
option( VIAME_ENABLE_DOCS        "Enable Documentation"     OFF )
option( VIAME_ENABLE_VIVIA       "Enable Vivia GUIs"        OFF )
option( VIAME_ENABLE_KWANT       "Enable KW Scoring Code"   OFF )
option( VIAME_ENABLE_SCALLOP_TK  "Enable ScallopTK plugins" OFF )

###
# Other misc options
##
option( VIAME_DISABLE_GPU_SUPPORT     "Force the disabling of all GPU support"  OFF )
option( VIAME_DISABLE_FFMPEG_SUPPORT  "Force the disabling of FFMPEG support"   OFF )
option( VIAME_DISABLE_CUDNN_SUPPORT   "Force the disabling of cuDNN support"    OFF )
mark_as_advanced( VIAME_DISABLE_CUDNN_SUPPORT )

###
# Add external package build instructions
##
option( VIAME_BUILD_DEPENDENCIES  "Build all required dependencies in a super-build" ON )

###
# Add logic relating to enable flags
##
if( VIAME_ENABLE_MATLAB )
  find_package( Matlab REQUIRED COMPONENTS ENG_LIBRARY MX_LIBRARY )

  get_filename_component( Matlab_LIBRARY_DIR "${Matlab_ENG_LIBRARY}" DIRECTORY )
  set( SET_MATLAB_LD_LIBRARY_PATH "export LD_LIBRARY_PATH=${Matlab_LIBRARY_DIR}:$LD_LIBRARY_PATH" )
endif()

if( WIN32 AND VIAME_ENABLE_CAFFE )
  message( FATAL_ERROR "Internal Caffe build on windows not currently supported, \
    please build Caffe externally using the Caffe windows branch, if desired." )
endif()

if( VIAME_ENABLE_CAFFE OR VIAME_ENABLE_SCALLOP_TK )
  set( VIAME_ENABLE_OPENCV ON CACHE BOOL "OpenCV required for other projects" FORCE )
endif()

if( VIAME_ENABLE_VIVIA )
  set( VIAME_ENABLE_VXL ON CACHE BOOL "VXL required for other projects" FORCE )
endif()

###
# Main build loop for when we are building both VIAME and all dependencies
##
if( VIAME_BUILD_DEPENDENCIES )

# Macros and scripts
include( ExternalProject )
include( common_args )
include( common_macros )

# Hard VIAME requirements
include( add_project_fletch )
include( add_project_kwiver )

# Optional requirements
if( VIAME_ENABLE_SCALLOP_TK )
  include( add_project_scallop_tk )
endif()

if( VIAME_ENABLE_VIVIA )
  include( add_project_vibrant )
  include( add_project_vivia )
endif()

if( VIAME_ENABLE_KWANT )
  include( add_project_kwant )
endif()

# Viame plugins
include( add_project_viame )

else()

###
# Look for and use Fletch to find dependencies
##
find_package( kwiver REQUIRED )
include_directories( ${KWIVER_INCLUDE_DIRS} )
link_directories( ${KWIVER_LIBRARY_DIR} )
list( INSERT CMAKE_MODULE_PATH  0         "${KWIVER_CMAKE_DIR}" )

if( IS_DIRECTORY ${fletch_DIR} )
  find_package( fletch NO_MODULE )
  list( APPEND CMAKE_PREFIX_PATH  "${fletch_DIR}" )
endif()

if( VIAME_ENABLE_OPENCV )
  find_package( OpenCV 2.4.6 REQUIRED )
endif()

include_directories( "${CMAKE_CURRENT_BINARY_DIR}" "${KWIVER_INCLUDE_DIRS}" )

###
# KWIVER CMake config
##
include( kwiver-utils )

##
# System specific compiler flags
include( viame-flags )

##
# check compiler support
include( kwiver-configcheck )

# set the name for our package exports
set( kwiver_export_name      viame_exports )

###
# build submodules

###
# add subdirs
##
add_subdirectory( plugins )
add_subdirectory( pipelines )
add_subdirectory( examples )

###
# Configure setup script
##
set( VIAME_SETUP_SCRIPT_FILE    "${VIAME_BINARY_DIR}/setup_viame.sh" )

configure_file(
  ${CMAKE_SOURCE_DIR}/CMake/setup_viame.sh.in
  ${VIAME_SETUP_SCRIPT_FILE}
  )

install( PROGRAMS   ${VIAME_SETUP_SCRIPT_FILE}
  DESTINATION       ${CMAKE_INSTALL_PREFIX} )

endif()
