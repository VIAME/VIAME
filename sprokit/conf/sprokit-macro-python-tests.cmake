# Should these functions be rectified with the kwiver versions
# in kwiver/CMake/utils/kwiver-utils-python-tests.cmake ?


###
#
# Configures the python test file to the ctest bin directory
#
# Args:
#     group: the suffix of the python file.
#     input: filename of the test .py file (includes the extension)
#
# SeeAlso:
#     sprokit-macro-tests.cmake
#     sprokit-macro-python.cmake
#     sprokit-macro-configure.cmake
#     ../support/test.cmake
#
function (sprokit_build_python_test group input)
  if (CMAKE_CONFIGURATION_TYPES)
    set(sprokit_configure_cmake_args
      "\"-Dconfig=${CMAKE_CFG_INTDIR}/\"")
  endif ()
  sprokit_configure_file(test-python-${group}
    "${CMAKE_CURRENT_SOURCE_DIR}/${input}"
    "${sprokit_test_output_path}/\${config}test-python-${group}"
    PYTHON_EXECUTABLE)

  sprokit_declare_tooled_test(python-${group})
endfunction ()


###
# Calls CMake `add_test` function under the hood
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
# separate `ctest` for each. Ideally we would just map the output from
# something like `py.test` to `ctest` instead.
#
# Arg:
#     group: the test is registered with this ctests group
#     file: filename of the test .py file (includes the extension)
#
# SeeAlso:
#     kwiver/CMake/utils/kwiver-utils-python-tests.cmake - defines kwiver_discover_python_tests
#     kwiver/sprokit/tests/bindings/python/sprokit/pipeline/CMakeLists.txt - uses this function
#
function (sprokit_discover_python_tests group file)
  file(STRINGS "${file}" test_lines)
  set(properties)

  sprokit_build_python_test("${group}" "${file}")

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
