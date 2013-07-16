project(sprokit_tests_test)

set(test_libraries
  sprokit_pipeline)

sprokit_build_tooled_test(test test_libraries test_test.cxx)

sprokit_add_tooled_test(test return_code)
sprokit_add_tooled_test(test error_string)
sprokit_add_tooled_test(test error_string_mid)
sprokit_add_tooled_test(test error_string_stdout)
sprokit_add_tooled_test(test error_string_second_line)
sprokit_add_tooled_test(test expected_exception)
sprokit_add_tooled_test(test unexpected_exception)
set(sprokit_test_environment TEST_ENVVAR=test_value)
sprokit_add_tooled_test(test environment)
set(sprokit_test_environment)

set_tests_properties(test-test-return_code
  PROPERTIES
    WILL_FAIL TRUE)

set_tests_properties(test-test-error_string
  PROPERTIES
    WILL_FAIL TRUE)

set_tests_properties(test-test-error_string_stdout
  PROPERTIES
    WILL_FAIL TRUE)

set_tests_properties(test-test-error_string_second_line
  PROPERTIES
    WILL_FAIL TRUE)

set_tests_properties(test-test-unexpected_exception
  PROPERTIES
    WILL_FAIL TRUE)

set(test_libraries
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${CMAKE_DL_LIBS})

sprokit_build_tooled_test(dlopen test_libraries test_dlopen.cxx)

foreach (library IN LISTS sprokit_libraries)
  sprokit_add_tooled_test(dlopen ${library}
    "$<TARGET_FILE:${library}>")
endforeach ()
