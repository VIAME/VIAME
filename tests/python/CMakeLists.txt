project(vistk_test_python)

find_package(PythonInterp REQUIRED)

set(TEST_RUNNER
  ${PYTHON_EXECUTABLE})

function (vistk_make_python_test group instance)
  set(python_module_path
    ${PYTHON_OUTPUT_PATH})

  if (WIN32)
    set(python_module_path
      "${python_module_path}/$<CONFIGURATION>")
  endif (WIN32)

  vistk_make_test(${group} ${instance}
    ${python_module_path} ${ARGN})
endfunction (vistk_make_python_test)
