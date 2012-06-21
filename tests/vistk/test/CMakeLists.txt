project(vistk_tests_test)

set(libraries)

vistk_build_test(test libraries test_test.cxx)

vistk_make_test(test return_code)
vistk_make_test(test error_string)
vistk_make_test(test error_string_mid)
vistk_make_test(test error_string_stdout)
vistk_make_test(test error_string_second_line)
vistk_make_test(test expected_exception)
vistk_make_test(test unexpected_exception)

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
  ${CMAKE_DL_LIBS})

vistk_build_test(dlopen libraries test_dlopen.cxx)

foreach (library ${vistk_libraries})
  vistk_make_test(dlopen ${library}
    "$<TARGET_FILE:${library}>")
endforeach (library)
