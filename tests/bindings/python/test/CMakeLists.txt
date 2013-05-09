project(sprokit_test_python_test)

sprokit_build_python_test(test test-test.py)

sprokit_add_python_test(test return_code)
sprokit_add_python_test(test error_string)
sprokit_add_python_test(test error_string_mid)
sprokit_add_python_test(test error_string_stdout)
sprokit_add_python_test(test error_string_second_line)
sprokit_add_python_test(test expected_exception)
sprokit_add_python_test(test unexpected_exception)
set(sprokit_test_environment
  TEST_ENVVAR=test_value)
sprokit_add_python_test(test environment)
set(sprokit_test_environment)

set_tests_properties(test-python-test-return_code
  PROPERTIES
    WILL_FAIL TRUE)

set_tests_properties(test-python-test-error_string
  PROPERTIES
    WILL_FAIL TRUE)

set_tests_properties(test-python-test-error_string_stdout
  PROPERTIES
    WILL_FAIL TRUE)

set_tests_properties(test-python-test-error_string_second_line
  PROPERTIES
    WILL_FAIL TRUE)

set_tests_properties(test-python-test-unexpected_exception
  PROPERTIES
    WILL_FAIL TRUE)
