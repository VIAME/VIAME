#
# Script to set up testing environment for kwiver components
#

set(no_install TRUE)

# GoogleTest is built by `third_party/googletest`, added by the top-level
# CMakeLists when tests are enabled. It was `find_package( GTest REQUIRED )`,
# resolved through fletch's prefix, and was the last thing in VIAME's own
# build that needed fletch at all.
if( NOT TARGET GTest::gtest )
  message( FATAL_ERROR "third_party/googletest has not been added yet" )
endif()

if (WIN32)

  # TODO: Output to a different directory and then use $<CONFIGURATION> in the
  # working path when generator expressions are supported in test properties.
  set(kwiver_test_output_path    "${KWIVER_BINARY_DIR}/bin")

else ()

  set(kwiver_test_output_path    "${KWIVER_BINARY_DIR}/tests/bin")
  set(kwiver_test_working_path    "${KWIVER_BINARY_DIR}/tests")

endif ()

# This sets the data directory relative to the current "tests" directory
set(kwiver_test_data_directory  "${KWIVER_SOURCE_DIR}/test_data")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
include_directories("${KWIVER_SOURCE_DIR}")
include_directories("${KWIVER_BINARY_DIR}")
include_directories("${KWIVER_SOURCE_DIR}/tests")
