project(sprokit_test_python)

find_package(PythonInterp ${PYTHON_VERSION} REQUIRED)

set(test_runner
  "${PYTHON_EXECUTABLE}")

function (sprokit_build_python_test group input)
  sprokit_configure_file(test-python-${group}
    "${CMAKE_CURRENT_SOURCE_DIR}/${input}"
    "${test_output_path}/test-python-${group}"
    PYTHON_EXECUTABLE)

  sprokit_declare_test(python-${group})
endfunction ()

function (sprokit_make_python_test group instance)
  set(python_module_path
    "${python_output_path}/${CMAKE_CFG_INTDIR}")
  set(python_chdir
    "${CMAKE_CFG_INTDIR}")

  sprokit_make_test(python-${group} ${instance}
    "${python_chdir}" "${python_module_path}" ${ARGN})
endfunction ()

add_subdirectory(test)

add_subdirectory(sprokit)
add_subdirectory(modules)
