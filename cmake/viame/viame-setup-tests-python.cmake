#
# Script to set up environment for python pytest
#
# To allow for windows configuration based paths in the build
# directory WIN_TEST_CONFIG_TYPE must be set to, in some fashion,
# evaluate to the configuration type of the build
# Note, this only need be set for Windows


# results of tests being run will be exported to an Xunit xml file
if (KWIVER_ENABLE_PYTHON_TESTS)

  set(no_install TRUE)
  string(TOLOWER "${CMAKE_PROJECT_NAME}" project_name)

  if (WIN32)
    set(kwiver_test_output_path    "${KWIVER_BINARY_DIR}/${WIN_TEST_CONFIG_TYPE}/bin")
  else ()
    set(kwiver_test_output_path    "${KWIVER_BINARY_DIR}/tests/bin")
  endif ()

  if (CTEST_BINARY_DIRECTORY)
    set(kwiver_test_working_path "${CTEST_BINARY_DIRECTORY}")
  else()
    set(kwiver_test_working_path "${KWIVER_BINARY_DIR}")
  endif()
  get_python_mod_dst()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(kwiver_pytest_runner "${Python3_EXECUTABLE}" -m pytest)
endif()
