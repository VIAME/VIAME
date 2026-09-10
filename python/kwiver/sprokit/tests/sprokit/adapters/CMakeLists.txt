project(sprokit_test_python_sprokit_adapters)

# For pipelines that also can be used by C++ tests
set(cpp_pipelines_dir
  "${KWIVER_SOURCE_DIR}/sprokit/tests/data/pipelines")

# For pipelines that are meant to be tested with Python
set(py_pipelines_dir
  "${CMAKE_CURRENT_SOURCE_DIR}/pipelines/")


# Simple testables require no pytest paramatarization
set(__simple_testable_modnames
  adapter_data_set
  )


###
# Setup simple tests with no parameters

# KWIVER_ENABLE_TESTS is always true for now (because we are in the test
# subdir), but the python tests should be moved to the python source dir.
if (KWIVER_ENABLE_TESTS)
  include(kwiver-test-setup)
  # Register individual tests with ctest
  foreach (group IN LISTS __simple_testable_modnames)
    # SeeAlso:
    #     # defines sprokit_discover_python_tests
    #     sprokit/conf/sprokit-macro-python-tests.cmake
    kwiver_discover_python_tests(${group} test-${group}.py)
  endforeach()
endif()
###

###
# Simple testables require no pytest paramatarization
set(__paramaterized_testable_modnames
  embedded_pipeline
  )


# Setup simple tests with parameters
if (KWIVER_ENABLE_TESTS)
  include(kwiver-test-setup)
  foreach (group IN LISTS __paramaterized_testable_modnames)
    kwiver_discover_python_tests(${group} test-${group}.py
      ${cpp_pipelines_dir}
      ${py_pipelines_dir})
  endforeach()
endif()
###
