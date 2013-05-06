project(sprokit_tests_test)

set(libraries
  sprokit_pipeline)

sprokit_build_test(test libraries test_test.cxx)

sprokit_make_test(test return_code)
sprokit_make_test(test error_string)
sprokit_make_test(test error_string_mid)
sprokit_make_test(test error_string_stdout)
sprokit_make_test(test error_string_second_line)
sprokit_make_test(test expected_exception)
sprokit_make_test(test unexpected_exception)
set(test_environment
  TEST_ENVVAR=test_value)
sprokit_make_test(test environment)
set(test_environment)

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

set(libraries
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${CMAKE_DL_LIBS})

sprokit_build_test(dlopen libraries test_dlopen.cxx)

foreach (library ${sprokit_libraries})
  sprokit_make_test(dlopen ${library}
    "$<TARGET_FILE:${library}>")
endforeach ()
