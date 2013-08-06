project(sprokit_test_python)

find_package(PythonInterp ${PYTHON_VERSION} REQUIRED)

set(sprokit_test_runner
  "${PYTHON_EXECUTABLE}")

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

function (sprokit_add_python_test group instance)
  set(python_module_path
    "${sprokit_python_output_path}")
  set(python_chdir
    ".")

  if (CMAKE_CONFIGURATION_TYPES)
    set(python_module_path
      "${python_module_path}/$<CONFIGURATION>")
    set(python_chdir
      "$<CONFIGURATION>")
  endif ()

  sprokit_add_tooled_test(python-${group} ${instance}
    "${python_chdir}" "${python_module_path}" ${ARGN})
endfunction ()

add_subdirectory(test)

add_subdirectory(sprokit)
add_subdirectory(modules)
