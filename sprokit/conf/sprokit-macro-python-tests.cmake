# Defines functions for registering python tests
# The following functions are defined:
#
#     sprokit_build_python_test
#     sprokit_add_python_test
#     sprokit_discover_python_tests
#
# The following variables may be used to control the behavior of the functions:
#
#     The same variables described in ./sprokit-macro-tests.cmake
#         of these `sprokit_test_runner` should be set to some command that
#         runs a python script in
#         kwiver/sprokit/tests/bindings/python/CMakeLists.txt
#
# For specific usage see documentation bellow


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
#     sprokit-macro-tests.cmake
#     sprokit-macro-python.cmake
#     sprokit-macro-configure.cmake
#     ..cmake/support/test.cmake
#
function (sprokit_build_python_test group input)
  if (CMAKE_CONFIGURATION_TYPES)
    set(sprokit_configure_cmake_args
      "\"-Dconfig=${CMAKE_CFG_INTDIR}/\"")
  endif ()

  set(name test-python-${group})
  set(source "${CMAKE_CURRENT_SOURCE_DIR}/${input}")
  set(dest "${sprokit_test_output_path}/\${config}test-python-${group}")

  if (KWIVER_SYMLINK_PYTHON)
    sprokit_symlink_file(${name} ${source} ${dest} PYTHON_EXECUTABLE)
  else()
    sprokit_configure_file(${name} ${source} ${dest} PYTHON_EXECUTABLE)
  endif()
  sprokit_declare_tooled_test(python-${group})
endfunction ()


###
# Registers a "built" python test with ctest
#
# Arg:
#     group: a group key previously registered with `sprokit_build_python_test`
#     instance: the name of the function to be tested.
#
# In most cases you should call sprokit_discover_python_tests instead.  The
# only exception is if you must paramaterizations via the commandline.
# Currently only `test-run` does this. In the future this will be removed in
# favor of pytest paramaterization.
#
# SeeAlso:
#     sprokit/tests/bindings/python/sprokit/pipeline/CMakeLists.txt - uses this func
#
function (sprokit_add_python_test group instance)
  set(python_module_path    "${sprokit_python_output_path}/${kwiver_python_subdir}")
  set(python_chdir          ".")

  if (CMAKE_CONFIGURATION_TYPES)
    set(python_module_path      "${sprokit_python_output_path}/$<CONFIGURATION>/${kwiver_python_subdir}")
    set(python_chdir           "$<CONFIGURATION>")
  endif ()

  sprokit_add_tooled_test(python-${group} ${instance}
    "${python_chdir}" "${python_module_path}/${python_sitename}" ${ARGN})
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
function (sprokit_discover_python_tests group file)
  file(STRINGS "${file}" test_lines)
  set(properties)

  sprokit_build_python_test("${group}" "${file}")

  # NOTE: most of this logic can be replaced by
  # `parse_python_testables` when PR #302 lands
  foreach (test_line IN LISTS test_lines)
    set(test_name)
    set(property)

    string(REGEX MATCH "^def test_([A-Za-z_]+)\\(.*\\):$"
      match "${test_line}")
    if (match)
      set(test_name "${CMAKE_MATCH_1}")
      sprokit_add_python_test("${group}" "${test_name}"
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
        set(sprokit_test_environment
          "${prop_value}")
      else ()
        set(property "${prop}" "${prop_value}")
        list(APPEND properties
          "${property}")
      endif ()
    endif ()
  endforeach ()
endfunction ()

