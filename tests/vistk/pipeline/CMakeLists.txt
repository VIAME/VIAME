project(vistk_test_pipeline)

set(libraries
  vistk_pipeline)

##############################
# Config tests
##############################
vistk_build_test(config libraries test_config.cxx)

vistk_make_test(config has_value)
vistk_make_test(config get_value)
vistk_make_test(config get_value_no_exist)
vistk_make_test(config get_value_type_mismatch)
vistk_make_test(config bool_conversion)
vistk_make_test(config unset_value)
vistk_make_test(config available_values)
vistk_make_test(config read_only)
vistk_make_test(config read_only_unset)
vistk_make_test(config subblock)
vistk_make_test(config subblock_view)
vistk_make_test(config merge_config)

##############################
# Datum tests
##############################
vistk_build_test(datum libraries test_datum.cxx)

vistk_make_test(datum empty)
vistk_make_test(datum complete)
vistk_make_test(datum error)
vistk_make_test(datum new)

##############################
# Dtor registry tests
##############################
vistk_build_test(dtor_registry libraries test_dtor_registry.cxx)

vistk_make_test(dtor_registry get_twice)
vistk_make_test(dtor_registry null_dtor)
vistk_make_test(dtor_registry call_dtor)

set_tests_properties(test-dtor_registry-call_dtor
  PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS: Dtor called")

vistk_make_test(dtor_registry module_marking)

##############################
# Stamp tests
##############################
vistk_build_test(stamp libraries test_stamp.cxx)

vistk_make_test(stamp coloring)
vistk_make_test(stamp equality)
vistk_make_test(stamp ordering)

##############################
# Edge tests
##############################
vistk_build_test(edge libraries test_edge.cxx)

vistk_make_test(edge null_config)
vistk_make_test(edge makes_dependency)
vistk_make_test(edge new_has_no_data)
vistk_make_test(edge new_is_not_full)
vistk_make_test(edge new_has_count_zero)
vistk_make_test(edge push_datum)
vistk_make_test(edge peek_datum)
vistk_make_test(edge pop_datum)
vistk_make_test(edge get_datum)
vistk_make_test(edge null_upstream_process)
vistk_make_test(edge null_downstream_process)
vistk_make_test(edge set_upstream_process)
vistk_make_test(edge set_downstream_process)
vistk_make_test(edge push_data_into_complete)
vistk_make_test(edge get_data_from_complete)

##############################
# Modules tests
##############################
add_library(processes_test SHARED
  processes_test.cxx)
target_link_libraries(processes_test
  vistk_pipeline)

if (WIN32)
  set(module_output RUNTIME_OUTPUT_DIRECTORY)
else (WIN32)
  set(module_output LIBRARY_OUTPUT_DIRECTORY)
endif (WIN32)

set_target_properties(processes_test
  PROPERTIES
    ${module_output} "${CMAKE_CURRENT_BINARY_DIR}"
    PREFIX           "")

vistk_build_test(modules libraries test_modules.cxx)

vistk_make_test(modules load)
vistk_make_test(modules multiple_load)
set(test_environment "VISTK_MODULE_PATH=${CMAKE_CURRENT_BINARY_DIR}")
vistk_make_test(modules envvar)
unset(test_environment)

##############################
# Process tests
##############################
vistk_build_test(process libraries test_process.cxx)

vistk_make_test(process null_input_edge)
vistk_make_test(process null_output_edge)
vistk_make_test(process connect_after_init)
vistk_make_test(process reinit)
vistk_make_test(process step_before_init)
vistk_make_test(process set_static_input_type)
vistk_make_test(process set_static_output_type)
vistk_make_test(process set_input_type_duplicate)
vistk_make_test(process set_output_type_duplicate)
vistk_make_test(process set_input_type_after_init)
vistk_make_test(process set_output_type_after_init)
vistk_make_test(process set_tagged_flow_dependent_port)
vistk_make_test(process set_tagged_flow_dependent_port_cascade)
vistk_make_test(process add_input_port_after_type_pin)
vistk_make_test(process add_output_port_after_type_pin)
vistk_make_test(process set_untagged_flow_dependent_port)
vistk_make_test(process null_config)
vistk_make_test(process null_input_port_info)
vistk_make_test(process null_output_port_info)
vistk_make_test(process null_conf_info)

##############################
# Schedule tests
##############################
vistk_build_test(schedule libraries test_schedule.cxx)

vistk_make_test(schedule null_config)
vistk_make_test(schedule null_pipeline)

##############################
# Pipeline tests
##############################
vistk_build_test(pipeline libraries test_pipeline.cxx)

vistk_make_test(pipeline null_config)
vistk_make_test(pipeline null_process)
vistk_make_test(pipeline add_process)
vistk_make_test(pipeline add_group)
vistk_make_test(pipeline duplicate_process_process)
vistk_make_test(pipeline duplicate_process_group)
vistk_make_test(pipeline duplicate_group_process)
vistk_make_test(pipeline duplicate_group_group)
vistk_make_test(pipeline map_input_no_group)
vistk_make_test(pipeline map_output_no_group)
vistk_make_test(pipeline map_input_no_process)
vistk_make_test(pipeline map_output_no_process)
vistk_make_test(pipeline map_input)
vistk_make_test(pipeline map_output)
vistk_make_test(pipeline connect_no_upstream)
vistk_make_test(pipeline connect_no_downstream)
vistk_make_test(pipeline connect_untyped_data_connection)
vistk_make_test(pipeline connect_untyped_flow_connection)
vistk_make_test(pipeline connect_type_force_data_upstream)
vistk_make_test(pipeline connect_type_force_data_upstream_reject)
vistk_make_test(pipeline connect_type_force_flow_upstream)
vistk_make_test(pipeline connect_type_force_flow_upstream_reject)
vistk_make_test(pipeline connect_type_force_flow_downstream)
vistk_make_test(pipeline connect_type_force_flow_downstream_reject)
vistk_make_test(pipeline connect_type_force_cascade_up)
vistk_make_test(pipeline connect_type_force_cascade_down)
vistk_make_test(pipeline connect_type_force_cascade_both)
vistk_make_test(pipeline connect_type_force_cascade_data_dependent)
vistk_make_test(pipeline connect_type_force_cascade_reject)
vistk_make_test(pipeline connect_type_mismatch)
vistk_make_test(pipeline connect_flag_mismatch)
vistk_make_test(pipeline connect)
vistk_make_test(pipeline connect_input_map)
vistk_make_test(pipeline connect_output_map)
vistk_make_test(pipeline setup_pipeline_no_processes)
vistk_make_test(pipeline setup_pipeline_orphaned_process)
vistk_make_test(pipeline setup_pipeline_backwards_edge)
vistk_make_test(pipeline setup_pipeline_not_a_dag)
vistk_make_test(pipeline setup_pipeline_data_dependent_set)
vistk_make_test(pipeline setup_pipeline_data_dependent_set_reject)
vistk_make_test(pipeline setup_pipeline_data_dependent_set_cascade)
vistk_make_test(pipeline setup_pipeline_data_dependent_set_cascade_reject)
vistk_make_test(pipeline setup_pipeline_untyped_data_dependent_unconnected)
vistk_make_test(pipeline setup_pipeline_untyped_data_dependent)
vistk_make_test(pipeline setup_pipeline_untyped_connection)
vistk_make_test(pipeline setup_pipeline_missing_required_input_connection)
vistk_make_test(pipeline setup_pipeline_missing_required_output_connection)
vistk_make_test(pipeline setup_pipeline_missing_required_group_input_connection)
vistk_make_test(pipeline setup_pipeline_missing_required_group_output_connection)
vistk_make_test(pipeline setup_pipeline_duplicate)
vistk_make_test(pipeline setup_pipeline_add_process)
vistk_make_test(pipeline setup_pipeline_add_group)
vistk_make_test(pipeline setup_pipeline_connect)
vistk_make_test(pipeline setup_pipeline_map_input)
vistk_make_test(pipeline setup_pipeline_map_output)
vistk_make_test(pipeline setup_pipeline)

##############################
# Introspection tests
##############################
vistk_build_test(process_introspection libraries test_process_introspection.cxx)

vistk_make_test(process_introspection all)

##############################
# Process registry tests
##############################
vistk_build_test(process_registry libraries test_process_registry.cxx)

vistk_make_test(process_registry get_twice)
vistk_make_test(process_registry null_config)
vistk_make_test(process_registry load_processes)
vistk_make_test(process_registry null_ctor)
vistk_make_test(process_registry duplicate_types)
vistk_make_test(process_registry unknown_types)
vistk_make_test(process_registry module_marking)

##############################
# Schedule registry tests
##############################
vistk_build_test(schedule_registry libraries test_schedule_registry.cxx)

vistk_make_test(schedule_registry get_twice)
vistk_make_test(schedule_registry null_config)
vistk_make_test(schedule_registry null_pipeline)
vistk_make_test(schedule_registry load_schedules)
vistk_make_test(schedule_registry null_ctor)
vistk_make_test(schedule_registry duplicate_types)
vistk_make_test(schedule_registry unknown_types)
vistk_make_test(schedule_registry module_marking)

##############################
# Running tests
##############################
vistk_build_test(run libraries test_run.cxx)

set(schedules
  sync
  thread_per_process)

if (VISTK_ENABLE_PYTHON)
  set(schedules
    ${schedules}
    pythread_per_process)
endif (VISTK_ENABLE_PYTHON)

if (VISTK_ENABLE_PYTHON)
  set(python_module_path
    "${PYTHON_OUTPUT_PATH}")

  if (WIN32)
    set(python_module_path
      "${python_module_path}/$<CONFIGURATION>")
  endif (WIN32)

  set(cur_python_path
    "$ENV{PYTHONPATH}")

  if (cur_python_path)
    set(cur_python_path
      ":${cur_python_path}")
  endif (cur_python_path)
endif (VISTK_ENABLE_PYTHON)

function (vistk_make_run_test group instance)
  foreach (schedule ${schedules})
    if (VISTK_ENABLE_PYTHON)
      if (${schedule} STREQUAL "pythread_per_process")
        set(test_environment "PYTHONPATH=${python_module_path}${cur_python_path}")
      endif (${schedule} STREQUAL "pythread_per_process")
    endif (VISTK_ENABLE_PYTHON)

    vistk_make_test(${group} ${instance}-${schedule})

    unset(test_environment)
  endforeach (schedule ${schedules})
endfunction (vistk_make_run_test group instance)

vistk_make_run_test(run simple_pipeline)
vistk_make_run_test(run multiplier_pipeline)
