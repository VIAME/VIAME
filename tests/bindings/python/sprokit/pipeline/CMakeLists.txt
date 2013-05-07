project(sprokit_test_python_sprokit_pipeline)

##############################
# Config tests
##############################
sprokit_build_python_test(config test-config.py)

sprokit_add_python_test(config import)
sprokit_add_python_test(config create)
sprokit_add_python_test(config api_calls)
sprokit_add_python_test(config has_value)
sprokit_add_python_test(config get_value)
sprokit_add_python_test(config get_value_nested)
sprokit_add_python_test(config get_value_no_exist)
sprokit_add_python_test(config unset_value)
sprokit_add_python_test(config available_values)
sprokit_add_python_test(config read_only)
sprokit_add_python_test(config read_only_unset)
sprokit_add_python_test(config subblock)
sprokit_add_python_test(config subblock_view)
sprokit_add_python_test(config merge_config)
sprokit_add_python_test(config dict)

##############################
# Datum tests
##############################
sprokit_build_python_test(datum test-datum.py)

sprokit_add_python_test(datum import)
sprokit_add_python_test(datum new)
sprokit_add_python_test(datum empty)
sprokit_add_python_test(datum flush)
sprokit_add_python_test(datum complete)
sprokit_add_python_test(datum error)

##############################
# Edge tests
##############################
sprokit_build_python_test(edge test-edge.py)

sprokit_add_python_test(edge import)
sprokit_add_python_test(edge create)
sprokit_add_python_test(edge datum_create)
sprokit_add_python_test(edge api_calls)
sprokit_add_python_test(edge datum_api_calls)

##############################
# Modules tests
##############################
sprokit_build_python_test(modules test-modules.py)

sprokit_add_python_test(modules import)
sprokit_add_python_test(modules load)

##############################
# Pipeline tests
##############################
sprokit_build_python_test(pipeline test-pipeline.py)

sprokit_add_python_test(pipeline import)
sprokit_add_python_test(pipeline create)
sprokit_add_python_test(pipeline api_calls)

##############################
# Process tests
##############################
sprokit_build_python_test(process test-process.py)

sprokit_add_python_test(process import)
sprokit_add_python_test(process create)
sprokit_add_python_test(process api_calls)

##############################
# Process cluster tests
##############################
sprokit_build_python_test(process_cluster test-process_cluster.py)

sprokit_add_python_test(process_cluster import)
sprokit_add_python_test(process_cluster api_calls)

##############################
# Process registry tests
##############################
sprokit_build_python_test(process_registry test-process_registry.py)

sprokit_add_python_test(process_registry import)
sprokit_add_python_test(process_registry create)
sprokit_add_python_test(process_registry api_calls)
sprokit_add_python_test(process_registry register)
sprokit_add_python_test(process_registry register_cluster)
sprokit_add_python_test(process_registry wrapper_api)

##############################
# Scheduler tests
##############################
sprokit_build_python_test(scheduler test-scheduler.py)

sprokit_add_python_test(scheduler import)

##############################
# Scheduler registry tests
##############################
sprokit_build_python_test(scheduler_registry test-scheduler_registry.py)

sprokit_add_python_test(scheduler_registry import)
sprokit_add_python_test(scheduler_registry create)
sprokit_add_python_test(scheduler_registry api_calls)
sprokit_add_python_test(scheduler_registry register)
sprokit_add_python_test(scheduler_registry wrapper_api)

##############################
# Stamp tests
##############################
sprokit_build_python_test(stamp test-stamp.py)

sprokit_add_python_test(stamp import)
sprokit_add_python_test(stamp create)
sprokit_add_python_test(stamp api_calls)

##############################
# Utils tests
##############################
sprokit_build_python_test(utils test-utils.py)

sprokit_add_python_test(utils import)
sprokit_add_python_test(utils create)
sprokit_add_python_test(utils name_thread)

##############################
# Version tests
##############################
sprokit_build_python_test(version test-version.py)

sprokit_add_python_test(version import)
sprokit_add_python_test(version api_calls)

##############################
# Run tests
##############################
sprokit_build_python_test(run test-run.py)

set(schedulers
  pythread_per_process
  sync
  thread_per_process)

function (sprokit_add_python_run_test group instance)
  foreach (scheduler ${schedulers})
    sprokit_add_python_test(${group} ${instance}-${scheduler})
  endforeach ()
endfunction ()

sprokit_add_python_run_test(run python_to_python)
sprokit_add_python_run_test(run cpp_to_python)
sprokit_add_python_run_test(run python_to_cpp)
sprokit_add_python_run_test(run python_via_cpp)
