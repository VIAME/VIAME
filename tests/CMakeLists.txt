project(sprokit_tests)

include("${sprokit_source_dir}/cmake/support/test.cmake")

set(sprokit_test_data_directory
  "${CMAKE_CURRENT_SOURCE_DIR}/data")

add_subdirectory(bindings)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory(sprokit)
