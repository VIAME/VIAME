# Defines functions for registering python tests
# The following functions are defined:
#
#     kwiver_build_python_test
#     kwiver_add_python_test
#     kwiver_discover_python_tests
#
# The following variables may be used to control the behavior of the functions:
#
#     The same variables described in ./kwiver-utils-tests.cmake
#         of these `kwiver_test_runner` should be set to some command that
#         runs a python script in
#         py_kwiver/sprokit/tests/bindings/python/CMakeLists.txt
#
# For specific usage see documentation below


###
#
# Registers a python tests and configures it to the ctest bin directory.
#
# Args:
#     group: A key that will point to the input file.
#     input: filename of the test .py file
#
# Notes:
#     The input file is typically the group with the prefix `test-` and the
#     suffix `.py. In other words: `input="test-%s.py" % group`.
#
# SeeAlso:
#     kwiver-utils-tests.cmake
#     kwiver-utils-python.cmake
#
function (kwiver_build_python_test group input)
  if (CMAKE_CONFIGURATION_TYPES)
    set(kwiver_configure_cmake_args
      "\"-Dconfig=${CMAKE_CFG_INTDIR}/\"")
  endif ()

  set(name test-python-${group})
  set(source "${CMAKE_CURRENT_SOURCE_DIR}/${input}")
  set(dest "${kwiver_test_output_path}/\${config}test-python-${group}")

  if (KWIVER_SYMLINK_PYTHON)
    kwiver_symlink_file(${name} ${source} ${dest} PYTHON_EXECUTABLE)
  else()
    kwiver_configure_file(${name} ${source} ${dest} PYTHON_EXECUTABLE)
  endif()
  kwiver_declare_test(python-${group})
endfunction ()


###
# Registers a "built" python test with ctest
#
# Arg:
#     group: a group key previously registered with `kwiver_build_python_test`
#     instance: the name of the function to be tested.
#
# In most cases you should call kwiver_discover_python_tests instead.  The
# only exception is if you must paramaterizations via the commandline.
# Currently only `test-run` does this. In the future this will be removed in
# favor of pytest paramaterization.
#
#
function (kwiver_add_python_test group instance)
  kwiver_add_test(python-${group} ${instance} ${ARGN})
endfunction ()


###
#
# Searches test .py files for functions that begin with "test" and creates a
# separate `ctest` for each.
#
# Arg:
#     group: the test is registered with this ctests group
#     file: filename of the test .py file (includes the extension)
#
# Notes:
#     The `group` argument is typically the name of the module being tested
#     The `file` argument is the actual name of the python file to be tested, which
#     typically looks like `test-<group>.py`
#
# SeeAlso:
#     kwiver/CMake/utils/kwiver-utils-python-tests.cmake - defines kwiver_discover_python_tests
#     kwiver/sprokit/tests/bindings/python/sprokit/pipeline/CMakeLists.txt - uses this function
#
function (kwiver_discover_python_tests group file)
  file(STRINGS "${file}" test_lines)
  set(properties)

  kwiver_build_python_test("${group}" "${file}")

  # NOTE: most of this logic can be replaced by
  # `parse_python_testables` when PR #302 lands
  foreach (test_line IN LISTS test_lines)
    set(test_name)
    set(property)

    string(REGEX MATCH "^def test_([A-Za-z_]+)\\(.*\\):$"
      match "${test_line}")
    if (match)
      set(test_name "${CMAKE_MATCH_1}")
      kwiver_add_python_test("${group}" "${test_name}"
        ${ARGN})
      if (properties)
        set_tests_properties("test-python-${group}-${test_name}"
          PROPERTIES
            ${properties})
      endif ()
      set(properties)
    endif ()
    string(REGEX MATCHALL "^# TEST_PROPERTY\\(([A-Za-z_]+), (.*)\\)$"
      match "${test_line}")
    if (match)
      set(prop "${CMAKE_MATCH_1}")
      string(CONFIGURE "${CMAKE_MATCH_2}" prop_value
        @ONLY)
      if (prop STREQUAL "ENVIRONMENT")
        set(kwiver_test_environment
          "${prop_value}")
      else ()
        set(property "${prop}" "${prop_value}")
        list(APPEND properties
          "${property}")
      endif ()
    endif ()
  endforeach ()
endfunction ()

###
# Adds a python module testing suite run by nosetests
#
function (kwiver_add_nosetests name targ)
  if (WIN32)
    add_test(
      NAME    test-python-${name}
      COMMAND cmd /C "${NOSE_COMMAND} ${kwiver_test_runner}${name}.py --with-xunit\
                                --xunit-file=nose_results.xml"
              ${ARGN})
  else()
    add_test(
      NAME    test-python-${name}
      COMMAND bash -c "${NOSE_COMMAND} ${kwiver_test_runner}${name}.py --with-xunit\
                                --xunit-file=nose_results.xml"
              ${ARGN})
  endif()

  set_tests_properties(test-python-${name}
    PROPERTIES
      FAIL_REGULAR_EXPRESSION "^Error: ;\nError: ")
  if (kwiver_test_working_path)
    set_tests_properties(test-python-${name}
      PROPERTIES
      WORKING_DIRECTORY "${kwiver_test_working_path}")
  endif ()
  if (kwiver_test_environment)
    set_tests_properties(test-python-${name}
      PROPERTIES
      ENVIRONMENT "${kwiver_test_environment}")
  endif ()
  if (KWIVER_TEST_ADD_TARGETS)
    add_custom_target(test-python-${name})
    add_custom_command(
      TARGET  test-python-${name}
      COMMAND ${kwiver_test_environment}
              ${kwiver_test_runner}
              "${kwiver_test_output_path}"
              ${ARGN}
      WORKING_DIRECTORY
              "${kwiver_test_working_path}"
      COMMENT "Running test \"${name}\"")
      add_dependencies(${targ}
                       test-python-${name})
  endif ()
endfunction()


###
# Add test data to nosetest directory
#
function (kwiver_python_add_test_data file_name file_dst)

  if(SKBUILD)
    set ( install_path "${file_dst}/tests/data")
  else()
    set ( install_path "${file_dst}/tests/data")
  endif()

  file(COPY ${file_name} DESTINATION ${install_path})

endfunction()
