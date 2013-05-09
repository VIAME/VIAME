project(sprokit_tests)

set(sprokit_test_output_path
  "${sprokit_binary_dir}/tests/bin")
set(sprokit_test_working_path
  "${sprokit_binary_dir}/tests")

include("${sprokit_source_dir}/conf/sprokit-macro-tests.cmake")
include("${sprokit_source_dir}/cmake/support/test.cmake")

set(sprokit_test_data_directory
  "${CMAKE_CURRENT_SOURCE_DIR}/data")

add_subdirectory(bindings)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory(sprokit)
