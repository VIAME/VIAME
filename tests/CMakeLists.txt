# VIAME Tests
#
# This directory contains consolidated tests for the VIAME project.
# Tests are only built when VIAME_ENABLE_TESTS is ON.

project( viame_tests )

# Set KWIVER directories for test setup (required by kwiver-test-setup.cmake)
if( NOT KWIVER_SOURCE_DIR )
  set( KWIVER_SOURCE_DIR "${VIAME_PACKAGES_DIR}/kwiver" )
endif()
if( NOT KWIVER_BINARY_DIR )
  set( KWIVER_BINARY_DIR "${CMAKE_BINARY_DIR}" )
endif()

include( kwiver-test-setup )

add_subdirectory( plugins )
add_subdirectory( tools )
add_subdirectory( examples )
