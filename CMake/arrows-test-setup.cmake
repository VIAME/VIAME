#
# Script to set up testing environment
#

set(no_install TRUE)

# ==================================================================
###
# Require Boost if std::random is not available
#

if (NOT VITAL_USE_STD_RANDOM)

  # Required Boost external dependency
  if(WIN32)
    set(Boost_USE_STATIC_LIBS TRUE)
  endif()

  find_package(Boost 1.50)
  add_definitions(-DBOOST_ALL_NO_LIB)
  include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

endif()

if (WIN32)
  # TODO: Output to a different directory and then use $<CONFIGURATION> in the
  # working path when generator expressions are supported in test properties.
  set(kwiver_test_output_path     "${ARROWS_BINARY_DIR}/bin")
else ()
  set(kwiver_test_output_path     "${ARROWS_BINARY_DIR}/tests/bin")
  set(kwiver_test_working_path    "${ARROWS_BINARY_DIR}/tests")
endif ()

set(arrows_test_data_directory      "${CMAKE_CURRENT_SOURCE_DIR}/test_data")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
include_directories("${ARROWS_SOURCE_DIR}")
include_directories("${ARROWS_BINARY_DIR}")

# this sets the data directory relative to the current "tests" directory
include_directories("${ALGORITHMS_SOURCE_DIR}/tests") # to pick up test_common.h -> there may be a better place for this
