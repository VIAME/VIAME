project(vistk_test_python)

find_package(PythonInterp ${PYTHON_VERSION} REQUIRED)

set(test_runner
  "${PYTHON_EXECUTABLE}")

function (vistk_build_python_test group input)
  vistk_configure_file(test-python-${group}
    "${CMAKE_CURRENT_SOURCE_DIR}/${input}"
    "${test_output_path}/test-python-${group}"
    PYTHON_EXECUTABLE)

  vistk_declare_test(python-${group})
endfunction (vistk_build_python_test)

function (vistk_make_python_test group instance)
  set(python_module_path
    "${python_output_path}")
  set(python_chdir
    ".")

  if (WIN32)
    set(python_module_path
      "${python_module_path}/${CMAKE_CFG_INTDIR}")
    set(python_chdir
      "${CMAKE_CFG_INTDIR}")
  endif (WIN32)

  vistk_make_test(python-${group} ${instance}
    "${python_chdir}" "${python_module_path}" ${ARGN})
endfunction (vistk_make_python_test)

add_subdirectory(test)

add_subdirectory(vistk)
add_subdirectory(modules)
