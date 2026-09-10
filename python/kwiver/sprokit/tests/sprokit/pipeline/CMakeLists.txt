project(sprokit_test_python_sprokit_pipeline)

# Simple testables require no pytest paramatarization
set(__simple_testable_modnames
  datum
  edge
  pipeline
  process
  process_cluster
  process_registry
  scheduler
  scheduler_registry
  stamp
  utils
  version
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
# Setup tests with parameters

set(schedulers
  pythread_per_process
  sync
  thread_per_process)

kwiver_build_python_test(run test-run.py)
function (kwiver_add_python_run_test group instance)
  foreach (scheduler IN LISTS schedulers)
    kwiver_add_python_test(${group} ${instance}-${scheduler})

    set_tests_properties(test-python-${group}-${instance}-${scheduler}
      PROPERTIES
        TIMEOUT 5)
  endforeach ()
endfunction ()

kwiver_add_python_run_test(run python_to_python)
kwiver_add_python_run_test(run cpp_to_python)
kwiver_add_python_run_test(run python_to_cpp)
kwiver_add_python_run_test(run python_via_cpp)
