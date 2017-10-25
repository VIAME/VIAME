project(sprokit_test_python)

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


# Include the python helper functions previously defined here
include("${sprokit_source_dir}/conf/sprokit-macro-python-tests.cmake")


add_subdirectory(test)

add_subdirectory(sprokit)
add_subdirectory(modules)
