# VIAME Tests
#
# This directory contains consolidated tests for the VIAME project.
# Tests are only built when VIAME_ENABLE_TESTS is ON.

project( viame_tests )

include( viame-test-setup )

# Shared pytest -> ctest registration helper, used by examples and pipelines.
include( "${CMAKE_CURRENT_SOURCE_DIR}/common/add_pytest_test.cmake" )

add_subdirectory( baseline )
add_subdirectory( startup )
add_subdirectory( library )
add_subdirectory( golden )
add_subdirectory( plugins )
add_subdirectory( tools )
add_subdirectory( examples )
add_subdirectory( pipelines )
