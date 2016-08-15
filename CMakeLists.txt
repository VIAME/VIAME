###
# top level build file for NOAA VIAME
#

cmake_minimum_required(VERSION 3.0)

project(VIAME)

include(CMakeDependentOption)

###
# Look for and use Fletch to find dependencies
#
find_package( kwiver REQUIRED )

if ( IS_DIRECTORY ${fletch_DIR} )
  find_package( fletch NO_MODULE )
endif()

find_package( OpenCV 2.4.6 REQUIRED )

include_directories( "${CMAKE_CURRENT_BINARY_DIR}" "${KWIVER_INCLUDE_DIRS}" )

LIST( INSERT CMAKE_MODULE_PATH  0      "${CMAKE_CURRENT_SOURCE_DIR}/CMake" "${KWIVER_CMAKE_DIR}" )

###
# KWIVER CMake config
#
include(kwiver-utils)

##
# System specific compiler flags
include(viame-flags)

##
# check compiler support
include( kwiver-configcheck )

# set the name for our package exports
set(kwiver_export_name      viame_exports)

###
# build submodules


###
# add subdirs
## add_subdirectory( processes )
add_subdirectory( algorithms )
add_subdirectory( pipelines )
