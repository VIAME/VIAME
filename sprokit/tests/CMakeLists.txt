project(sprokit_tests)

set(no_install TRUE)

if (WIN32)
  # TODO: Output to a different directory and then use $<CONFIGURATION> in the
  # working path when generator expressions are supported in test properties.
  set(sprokit_test_output_path
    "${sprokit_output_dir}/bin")
else ()
  set(sprokit_test_output_path
    "${sprokit_binary_dir}/tests/bin")
  set(sprokit_test_working_path
    "${sprokit_binary_dir}/tests")
endif ()

include("${sprokit_source_dir}/conf/sprokit-macro-tests.cmake")
include("${sprokit_source_dir}/cmake/support/test.cmake")

set(sprokit_test_data_directory
  "${CMAKE_CURRENT_SOURCE_DIR}/data")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory(sprokit)
