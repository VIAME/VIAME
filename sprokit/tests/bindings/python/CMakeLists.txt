project(sprokit_test_python)

find_package(PythonInterp ${PYTHON_VERSION} REQUIRED)

# TODO: Use "$<CONFIGURATION>" and remove chdir below when generator
# expressions are supported in test properties.
if (WIN32)
  set(sprokit_test_working_path
    "${sprokit_output_dir}/bin")
endif ()

cmake_dependent_option(SPROKIT_ENABLE_COVERAGE_PYTHON "" ON
  SPROKIT_ENABLE_COVERAGE OFF)

if (SPROKIT_ENABLE_COVERAGE_PYTHON)
  set(sprokit_test_runner
    "${PYTHON_EXECUTABLE}"
    -m trace
    --count
    --coverdir "${sprokit_test_working_path}"
    --ignore-dir="\$prefix"
    --ignore-dir="\$exec_prefix")
else ()
  set(sprokit_test_runner
    "${PYTHON_EXECUTABLE}")
endif ()

###
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
#
function (sprokit_add_python_test group instance)
  set(python_module_path    "${sprokit_python_output_path}/${sprokit_python_subdir}")
  set(python_chdir          ".")

  if (CMAKE_CONFIGURATION_TYPES)
    set(python_module_path      "${sprokit_python_output_path}/$<CONFIGURATION>/${sprokit_python_subdir}")
    set(python_chdir           "$<CONFIGURATION>")
  endif ()

  _sprokit_python_site_package_dir( site_dir )
  sprokit_add_tooled_test(python-${group} ${instance}
    "${python_chdir}" "${python_module_path}/${site_dir}" ${ARGN})
endfunction ()

###
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

add_subdirectory(test)

add_subdirectory(sprokit)
add_subdirectory(modules)
