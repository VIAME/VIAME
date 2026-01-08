# VIAME Tests
#
# This directory contains consolidated tests for the VIAME project.
# Tests are only built when VIAME_ENABLE_TESTS is ON.

project( viame_tests )

include( kwiver-test-setup )

add_subdirectory( plugins )
add_subdirectory( tools )
add_subdirectory( examples )
