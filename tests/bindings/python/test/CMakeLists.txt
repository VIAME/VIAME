project(vistk_test_python_test)

vistk_build_python_test(test test-test.py)

vistk_make_python_test(test return_code)
vistk_make_python_test(test error_string)
vistk_make_python_test(test error_string_mid)
vistk_make_python_test(test error_string_stdout)
vistk_make_python_test(test error_string_second_line)
vistk_make_python_test(test expected_exception)
vistk_make_python_test(test unexpected_exception)
set(test_environment
  TEST_ENVVAR=test_value)
vistk_make_python_test(test environment)
set(test_environment)

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
