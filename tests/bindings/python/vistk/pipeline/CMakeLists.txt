project(vistk_test_python_vistk_pipeline)

##############################
# Config tests
##############################
vistk_build_python_test(config test-config.py)

vistk_make_python_test(config import)
vistk_make_python_test(config create)
vistk_make_python_test(config api_calls)
vistk_make_python_test(config has_value)
vistk_make_python_test(config get_value)
vistk_make_python_test(config get_value_nested)
vistk_make_python_test(config get_value_no_exist)
vistk_make_python_test(config unset_value)
vistk_make_python_test(config available_values)
vistk_make_python_test(config read_only)
vistk_make_python_test(config read_only_unset)
vistk_make_python_test(config subblock)
vistk_make_python_test(config subblock_view)
vistk_make_python_test(config merge_config)
vistk_make_python_test(config dict)

##############################
# Datum tests
##############################
vistk_build_python_test(datum test-datum.py)

vistk_make_python_test(datum import)
vistk_make_python_test(datum new)
vistk_make_python_test(datum empty)
vistk_make_python_test(datum flush)
vistk_make_python_test(datum complete)
vistk_make_python_test(datum error)

##############################
# Edge tests
##############################
vistk_build_python_test(edge test-edge.py)

vistk_make_python_test(edge import)
vistk_make_python_test(edge create)
vistk_make_python_test(edge datum_create)
vistk_make_python_test(edge api_calls)
vistk_make_python_test(edge datum_api_calls)

##############################
# Modules tests
##############################
vistk_build_python_test(modules test-modules.py)

vistk_make_python_test(modules import)
vistk_make_python_test(modules load)

##############################
# Pipeline tests
##############################
vistk_build_python_test(pipeline test-pipeline.py)

vistk_make_python_test(pipeline import)
vistk_make_python_test(pipeline create)
vistk_make_python_test(pipeline api_calls)

##############################
# Process tests
##############################
vistk_build_python_test(process test-process.py)

vistk_make_python_test(process import)
vistk_make_python_test(process create)
vistk_make_python_test(process api_calls)

##############################
# Process cluster tests
##############################
vistk_build_python_test(process_cluster test-process_cluster.py)

vistk_make_python_test(process_cluster import)
vistk_make_python_test(process_cluster api_calls)

##############################
# Process registry tests
##############################
vistk_build_python_test(process_registry test-process_registry.py)

vistk_make_python_test(process_registry import)
vistk_make_python_test(process_registry create)
vistk_make_python_test(process_registry api_calls)
vistk_make_python_test(process_registry register)
vistk_make_python_test(process_registry register_cluster)
vistk_make_python_test(process_registry wrapper_api)

##############################
# Scheduler tests
##############################
vistk_build_python_test(scheduler test-scheduler.py)

vistk_make_python_test(scheduler import)

##############################
# Scheduler registry tests
##############################
vistk_build_python_test(scheduler_registry test-scheduler_registry.py)

vistk_make_python_test(scheduler_registry import)
vistk_make_python_test(scheduler_registry create)
vistk_make_python_test(scheduler_registry api_calls)
vistk_make_python_test(scheduler_registry register)
vistk_make_python_test(scheduler_registry wrapper_api)

##############################
# Stamp tests
##############################
vistk_build_python_test(stamp test-stamp.py)

vistk_make_python_test(stamp import)
vistk_make_python_test(stamp create)
vistk_make_python_test(stamp api_calls)

##############################
# Utils tests
##############################
vistk_build_python_test(utils test-utils.py)

vistk_make_python_test(utils import)
vistk_make_python_test(utils create)
vistk_make_python_test(utils name_thread)

##############################
# Version tests
##############################
vistk_build_python_test(version test-version.py)

vistk_make_python_test(version import)
vistk_make_python_test(version api_calls)

##############################
# Run tests
##############################
vistk_build_python_test(run test-run.py)

set(schedulers
  pythread_per_process
  sync
  thread_per_process)

function (vistk_make_python_run_test group instance)
  foreach (scheduler ${schedulers})
    vistk_make_python_test(${group} ${instance}-${scheduler})
  endforeach (scheduler ${schedulers})
endfunction (vistk_make_python_run_test group instance)

vistk_make_python_run_test(run python_to_python)
vistk_make_python_run_test(run cpp_to_python)
vistk_make_python_run_test(run python_to_cpp)
vistk_make_python_run_test(run python_via_cpp)
