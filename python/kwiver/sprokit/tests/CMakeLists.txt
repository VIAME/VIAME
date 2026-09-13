project(sprokit_test_python)

# TODO: Use "$<CONFIGURATION>" and remove chdir below when generator
# expressions are supported in test properties.
include( viame-test-setup )

cmake_dependent_option(SPROKIT_ENABLE_COVERAGE_PYTHON "" ON
  SPROKIT_ENABLE_COVERAGE OFF)


if (SPROKIT_ENABLE_COVERAGE_PYTHON)
  string(TOLOWER "${CMAKE_PROJECT_NAME}" project_name)
  set(kwiver_test_runner
    "${Python_EXECUTABLE}"
    -m trace
    --count
    --coverdir "${python_install_path}/${kwiver_python_subdir}/${python_sitename}/${project_name}")
else ()
  set(kwiver_test_runner
    "${Python_EXECUTABLE}")
endif ()

add_subdirectory(test)
add_subdirectory(sprokit)
add_subdirectory(modules)
add_subdirectory(processes)

viame_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
                         sprokit/tests
                         __init__)
