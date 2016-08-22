###
# top level build file for NOAA VIAME
#

cmake_minimum_required( VERSION 3.0 )

project( VIAME )

include( CMakeDependentOption )

###
# Add paths used throughout CMakeLists files
##
LIST( INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/CMake" "${KWIVER_CMAKE_DIR}" )

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
option( VIAME_ENABLE_SCALLOP_TK  "Enable ScallopTK plugins" OFF )

###
# Add external package build instructions
##
option( VIAME_EXT_PROJECTS_ONLY  "Enable only external projects" ON )
mark_as_advanced( VIAME_EXT_PROJECTS_ONLY )

if( VIAME_EXT_PROJECTS_ONLY )

# Macros and scripts
include( ExternalProject )
include( common_args )

# Hard VIAME requirements
include( external_fletch )
include( external_kwiver )
include( external_viame )

# Optional requirements
if( VIAME_ENABLE_SCALLOP_TK )
  include( external_scallop_tk )
endif()

else()

###
# Look for and use Fletch to find dependencies
##
find_package( kwiver REQUIRED )
include_directories( ${KWIVER_INCLUDE_DIRS} )
link_directories( ${KWIVER_LIBRARY_DIR} )
LIST( INSERT CMAKE_MODULE_PATH  0         "${KWIVER_CMAKE_DIR}" )

if ( IS_DIRECTORY ${fletch_DIR} )
  find_package( fletch NO_MODULE )
  LIST( APPEND CMAKE_PREFIX_PATH  "${fletch_DIR}" )
endif()

find_package( OpenCV 2.4.6 REQUIRED )

include_directories( "${CMAKE_CURRENT_BINARY_DIR}" "${KWIVER_INCLUDE_DIRS}" )

###
# KWIVER CMake config
#
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
#
kwiver_configure_file(viame-buid-setup
  "${CMAKE_SOURCE_DIR}/CMake/setup_VIAME.sh.in"
  setup_VIAME.sh
  fletch_ROOT
  KWIVER_MODULE_DIR
  KWIVER_LIBRARY_DIR
  )

endif()
