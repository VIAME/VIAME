set(kwiver_test_pipelines_directory
  "${KWIVER_SOURCE_DIR}/sprokit/tests/data/pipelines")

set(__modnames
  bake
  export_
  load
  )

# For now this is always enabled (because we are in the test-dir)
# but the goal is to move python tests into the python source dir soon
if (KWIVER_ENABLE_TESTS)
  include(kwiver-test-setup)
  foreach (group IN LISTS __modnames)
    kwiver_discover_python_tests(${group} test-${group}.py
      ${kwiver_test_pipelines_directory})
  endforeach()
endif()
