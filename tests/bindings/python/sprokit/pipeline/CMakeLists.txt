project(sprokit_test_python_sprokit_pipeline)

##############################
# Config tests
##############################
sprokit_discover_python_tests(config test-config.py)

##############################
# Datum tests
##############################
sprokit_discover_python_tests(datum test-datum.py)

##############################
# Edge tests
##############################
sprokit_discover_python_tests(edge test-edge.py)

##############################
# Modules tests
##############################
sprokit_discover_python_tests(modules test-modules.py)

##############################
# Pipeline tests
##############################
sprokit_discover_python_tests(pipeline test-pipeline.py)

##############################
# Process tests
##############################
sprokit_discover_python_tests(process test-process.py)

##############################
# Process cluster tests
##############################
sprokit_discover_python_tests(process_cluster test-process_cluster.py)

##############################
# Process registry tests
##############################
sprokit_discover_python_tests(process_registry test-process_registry.py)

##############################
# Scheduler tests
##############################
sprokit_discover_python_tests(scheduler test-scheduler.py)

##############################
# Scheduler registry tests
##############################
sprokit_discover_python_tests(scheduler_registry test-scheduler_registry.py)

##############################
# Stamp tests
##############################
sprokit_discover_python_tests(stamp test-stamp.py)

##############################
# Utils tests
##############################
sprokit_discover_python_tests(utils test-utils.py)

##############################
# Version tests
##############################
sprokit_discover_python_tests(version test-version.py)

##############################
# Run tests
##############################
sprokit_build_python_test(run test-run.py)

set(schedulers
  pythread_per_process
  sync
  thread_per_process)

function (sprokit_add_python_run_test group instance)
  foreach (scheduler IN LISTS schedulers)
    sprokit_add_python_test(${group} ${instance}-${scheduler})

    set_tests_properties(test-python-${group}-${instance}-${scheduler}
      PROPERTIES
        TIMEOUT 5)
  endforeach ()
endfunction ()

sprokit_add_python_run_test(run python_to_python)
sprokit_add_python_run_test(run cpp_to_python)
sprokit_add_python_run_test(run python_to_cpp)
sprokit_add_python_run_test(run python_via_cpp)
