project(vistk_test_python)

find_package(PythonInterp REQUIRED)

set(TEST_RUNNER
  "${PYTHON_EXECUTABLE}")

function (vistk_build_python_test group input)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/${input}"
    "${EXECUTABLE_OUTPUT_PATH}/test-python-${group}"
    @ONLY)
  set_directory_properties(PROPERTIES
    ADDITIONAL_MAKE_CLEAN_FILES "${EXECUTABLE_OUTPUT_PATH}/test-python-${group}")

  vistk_declare_test(python-${group})
endfunction (vistk_build_python_test)

function (vistk_make_python_test group instance)
  set(python_module_path
    "${PYTHON_OUTPUT_PATH}")
  set(python_chdir
    ".")

  if (WIN32)
    set(python_module_path
      "${python_module_path}/$<CONFIGURATION>")
    set(python_chdir
      "$<CONFIGURATION>")
  endif (WIN32)

  vistk_make_test(python-${group} ${instance}
    "${python_chdir}" "${python_module_path}" ${ARGN})
endfunction (vistk_make_python_test)

add_subdirectory(image)
add_subdirectory(modules)
add_subdirectory(pipeline)
add_subdirectory(pipeline_util)
