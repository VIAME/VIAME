project(kwiver_tests)

set(no_install TRUE)

set(test_libraries kwiver kwiver_apm)

if (WIN32)
  # TODO: Output to a different directory and then use $<CONFIGURATION> in the
  # working path when generator expressions are supported in test properties.
  set(kwiver_test_output_path
    "${KWIVER_BINARY_DIR}/bin")
else ()
  set(kwiver_test_output_path
    "${KWIVER_BINARY_DIR}/tests/bin")
  set(kwiver_test_working_path
    "${KWIVER_BINARY_DIR}/tests")
endif ()


set(kwiver_test_data_directory
  "${CMAKE_CURRENT_SOURCE_DIR}/data")


include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
include_directories("${KWIVER_SOURCE_DIR}")
include_directories("${KWIVER_BINARY_DIR}")

add_subdirectory(kwiver)
