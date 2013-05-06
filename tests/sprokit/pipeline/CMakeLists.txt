project(sprokit_test_pipeline)

set(libraries
  sprokit_pipeline)

##############################
# Config tests
##############################
sprokit_build_test(config libraries test_config.cxx)

sprokit_make_test(config block_sep_size)
sprokit_make_test(config has_value)
sprokit_make_test(config get_value)
sprokit_make_test(config get_value_nested)
sprokit_make_test(config get_value_no_exist)
sprokit_make_test(config get_value_type_mismatch)
sprokit_make_test(config bool_conversion)
sprokit_make_test(config unset_value)
sprokit_make_test(config available_values)
sprokit_make_test(config read_only)
sprokit_make_test(config read_only_unset)
sprokit_make_test(config subblock)
sprokit_make_test(config subblock_nested)
sprokit_make_test(config subblock_match)
sprokit_make_test(config subblock_prefix_match)
sprokit_make_test(config subblock_view)
sprokit_make_test(config subblock_view_nested)
sprokit_make_test(config subblock_view_match)
sprokit_make_test(config subblock_view_prefix_match)
sprokit_make_test(config merge_config)

##############################
# Datum tests
##############################
sprokit_build_test(datum libraries test_datum.cxx)

sprokit_make_test(datum empty)
sprokit_make_test(datum flush)
sprokit_make_test(datum complete)
sprokit_make_test(datum error)
sprokit_make_test(datum new)

##############################
# Stamp tests
##############################
sprokit_build_test(stamp libraries test_stamp.cxx)

sprokit_make_test(stamp equality)
sprokit_make_test(stamp ordering)

##############################
# Edge tests
##############################
set(edge_libraries
  ${libraries}
  ${Boost_THREAD_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_CHRONO_LIBRARY})

sprokit_build_test(edge edge_libraries test_edge.cxx)

sprokit_make_test(edge null_config)
sprokit_make_test(edge makes_dependency)
sprokit_make_test(edge new_has_no_data)
sprokit_make_test(edge new_is_not_full)
sprokit_make_test(edge new_has_count_zero)
sprokit_make_test(edge push_datum)
sprokit_make_test(edge peek_datum)
sprokit_make_test(edge peek_datum_index)
sprokit_make_test(edge pop_datum)
sprokit_make_test(edge get_datum)
sprokit_make_test(edge null_upstream_process)
sprokit_make_test(edge null_downstream_process)
sprokit_make_test(edge set_upstream_process)
sprokit_make_test(edge set_downstream_process)
sprokit_make_test(edge push_data_into_complete)
sprokit_make_test(edge get_data_from_complete)
sprokit_make_test(edge capacity)

##############################
# Modules tests
##############################
add_library(processes_test SHARED
  processes_test.cxx)
target_link_libraries(processes_test
  LINK_PRIVATE
    sprokit_pipeline)

set_target_properties(processes_test
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    PREFIX                   ""
    SUFFIX                   ${CMAKE_SHARED_MODULE_SUFFIX})

sprokit_build_test(modules libraries test_modules.cxx)

sprokit_make_test(modules load)
sprokit_make_test(modules multiple_load)
set(test_environment "SPROKIT_MODULE_PATH=${CMAKE_CURRENT_BINARY_DIR}")
sprokit_make_test(modules envvar)
unset(test_environment)

##############################
# Process tests
##############################
sprokit_build_test(process libraries test_process.cxx)

sprokit_make_test(process null_input_edge)
sprokit_make_test(process null_output_edge)
sprokit_make_test(process connect_after_init)
sprokit_make_test(process configure_twice)
sprokit_make_test(process reinit)
sprokit_make_test(process reset)
sprokit_make_test(process step_before_configure)
sprokit_make_test(process step_before_init)
sprokit_make_test(process set_static_input_type)
sprokit_make_test(process set_static_output_type)
sprokit_make_test(process set_input_type_duplicate)
sprokit_make_test(process set_output_type_duplicate)
sprokit_make_test(process set_input_type_after_init)
sprokit_make_test(process set_output_type_after_init)
sprokit_make_test(process set_tagged_flow_dependent_port)
sprokit_make_test(process set_tagged_flow_dependent_port_cascade)
sprokit_make_test(process set_tagged_flow_dependent_port_cascade_any)
sprokit_make_test(process add_input_port_after_type_pin)
sprokit_make_test(process add_output_port_after_type_pin)
sprokit_make_test(process set_untagged_flow_dependent_port)
sprokit_make_test(process remove_input_port)
sprokit_make_test(process remove_output_port)
sprokit_make_test(process remove_non_exist_input_port)
sprokit_make_test(process remove_non_exist_output_port)
sprokit_make_test(process remove_only_tagged_flow_dependent_port)
sprokit_make_test(process remove_tagged_flow_dependent_port)
sprokit_make_test(process null_config)
sprokit_make_test(process null_input_port_info)
sprokit_make_test(process null_output_port_info)
sprokit_make_test(process null_conf_info)
sprokit_make_test(process tunable_config)
sprokit_make_test(process tunable_config_read_only)
sprokit_make_test(process reconfigure_tunable)
sprokit_make_test(process reconfigure_non_tunable)
sprokit_make_test(process reconfigure_extra_parameters)

##############################
# Process cluster tests
##############################
sprokit_build_test(process_cluster libraries test_process_cluster.cxx)

sprokit_make_test(process_cluster configure)
sprokit_make_test(process_cluster init)
sprokit_make_test(process_cluster step)
sprokit_make_test(process_cluster add_process)
sprokit_make_test(process_cluster duplicate_name)
sprokit_make_test(process_cluster map_config)
sprokit_make_test(process_cluster map_config_after_process)
sprokit_make_test(process_cluster map_config_no_exist)
sprokit_make_test(process_cluster map_config_read_only)
sprokit_make_test(process_cluster map_config_ignore_override)
sprokit_make_test(process_cluster map_input)
sprokit_make_test(process_cluster map_input_twice)
sprokit_make_test(process_cluster map_input_no_exist)
sprokit_make_test(process_cluster map_input_port_no_exist)
sprokit_make_test(process_cluster map_output)
sprokit_make_test(process_cluster map_output_twice)
sprokit_make_test(process_cluster map_output_no_exist)
sprokit_make_test(process_cluster map_output_port_no_exist)
sprokit_make_test(process_cluster connect)
sprokit_make_test(process_cluster connect_upstream_no_exist)
sprokit_make_test(process_cluster connect_downstream_no_exist)
sprokit_make_test(process_cluster connect_upstream_port_no_exist)
sprokit_make_test(process_cluster connect_downstream_port_no_exist)
sprokit_make_test(process_cluster reconfigure_pass_tunable_mappings)
sprokit_make_test(process_cluster reconfigure_no_pass_untunable_mappings)
sprokit_make_test(process_cluster reconfigure_pass_extra)
sprokit_make_test(process_cluster reconfigure_tunable_only_if_mapped)
sprokit_make_test(process_cluster reconfigure_mapped_untunable)

##############################
# Scheduler tests
##############################
sprokit_build_test(scheduler libraries test_scheduler.cxx)

sprokit_make_test(scheduler null_config)
sprokit_make_test(scheduler null_pipeline)
sprokit_make_test(scheduler start_scheduler)
sprokit_make_test(scheduler pause_scheduler)
sprokit_make_test(scheduler resume_scheduler)
sprokit_make_test(scheduler stop_scheduler)
sprokit_make_test(scheduler stop_paused_scheduler)
sprokit_make_test(scheduler restart_scheduler)
sprokit_make_test(scheduler repause_scheduler)
sprokit_make_test(scheduler pause_before_start_scheduler)
sprokit_make_test(scheduler wait_before_start_scheduler)
sprokit_make_test(scheduler stop_before_start_scheduler)
sprokit_make_test(scheduler resume_before_start_scheduler)
sprokit_make_test(scheduler resume_unpaused_scheduler)
sprokit_make_test(scheduler restart)

##############################
# Pipeline tests
##############################
sprokit_build_test(pipeline libraries test_pipeline.cxx)

sprokit_make_test(pipeline null_config)
sprokit_make_test(pipeline null_process)
sprokit_make_test(pipeline add_process)
sprokit_make_test(pipeline add_cluster)
sprokit_make_test(pipeline duplicate_process_process)
sprokit_make_test(pipeline connect_no_upstream)
sprokit_make_test(pipeline connect_no_downstream)
sprokit_make_test(pipeline connect_untyped_data_connection)
sprokit_make_test(pipeline connect_untyped_flow_connection)
sprokit_make_test(pipeline connect_type_mismatch)
sprokit_make_test(pipeline connect_flag_shared_no_mutate)
sprokit_make_test(pipeline connect_flag_mismatch_const_mutate)
sprokit_make_test(pipeline connect_flag_mismatch_shared_mutate_first)
sprokit_make_test(pipeline connect_flag_mismatch_shared_mutate_second)
sprokit_make_test(pipeline connect)
sprokit_make_test(pipeline setup_pipeline_no_processes)
sprokit_make_test(pipeline setup_pipeline_orphaned_process)
sprokit_make_test(pipeline setup_pipeline_type_force_flow_upstream)
sprokit_make_test(pipeline setup_pipeline_type_force_flow_downstream)
sprokit_make_test(pipeline setup_pipeline_type_force_cascade_up)
sprokit_make_test(pipeline setup_pipeline_type_force_cascade_down)
sprokit_make_test(pipeline setup_pipeline_type_force_cascade_both)
sprokit_make_test(pipeline setup_pipeline_backwards_edge)
sprokit_make_test(pipeline setup_pipeline_not_a_dag)
sprokit_make_test(pipeline setup_pipeline_data_dependent_set)
sprokit_make_test(pipeline setup_pipeline_data_dependent_set_reject)
sprokit_make_test(pipeline setup_pipeline_data_dependent_set_cascade)
sprokit_make_test(pipeline setup_pipeline_data_dependent_set_cascade_reject)
sprokit_make_test(pipeline setup_pipeline_type_force_flow_upstream_reject)
sprokit_make_test(pipeline setup_pipeline_type_force_flow_downstream_reject)
sprokit_make_test(pipeline setup_pipeline_type_force_cascade_reject)
sprokit_make_test(pipeline setup_pipeline_untyped_data_dependent)
sprokit_make_test(pipeline setup_pipeline_untyped_connection)
sprokit_make_test(pipeline setup_pipeline_missing_required_input_connection)
sprokit_make_test(pipeline setup_pipeline_missing_required_output_connection)
sprokit_make_test(pipeline setup_pipeline_duplicate)
sprokit_make_test(pipeline setup_pipeline_add_process)
sprokit_make_test(pipeline setup_pipeline_connect)
sprokit_make_test(pipeline setup_pipeline)
sprokit_make_test(pipeline start_before_setup)
sprokit_make_test(pipeline start_unsuccessful_setup)
sprokit_make_test(pipeline start_and_stop)
sprokit_make_test(pipeline reset_while_running)
sprokit_make_test(pipeline reset)
sprokit_make_test(pipeline remove_process)
sprokit_make_test(pipeline remove_process_after_setup)
sprokit_make_test(pipeline disconnect)
sprokit_make_test(pipeline disconnect_after_setup)
sprokit_make_test(pipeline reconfigure_before_setup)
sprokit_make_test(pipeline reconfigure)
sprokit_make_test(pipeline reconfigure_only_top_level)

##############################
# Introspection tests
##############################
sprokit_build_test(process_introspection libraries test_process_introspection.cxx)

if (SPROKIT_ENABLE_PYTHON)
  set(python_module_path
    "${python_output_path}/${CMAKE_CFG_INTDIR}")

  set(cur_python_path
    "$ENV{PYTHONPATH}")

  if (cur_python_path)
    set(cur_python_path
      ":${cur_python_path}")
  endif ()

  # TODO C++ exceptions get lost in the Python glue code.
  #set(test_environment "PYTHONPATH=${python_module_path}${cur_python_path}")
endif ()

sprokit_make_test(process_introspection all)

##############################
# Process registry tests
##############################
sprokit_build_test(process_registry libraries test_process_registry.cxx)

sprokit_make_test(process_registry get_twice)
sprokit_make_test(process_registry null_config)
sprokit_make_test(process_registry load_processes)
sprokit_make_test(process_registry null_ctor)
sprokit_make_test(process_registry duplicate_types)
sprokit_make_test(process_registry unknown_types)
sprokit_make_test(process_registry module_marking)
sprokit_make_test(process_registry register_cluster)

##############################
# Scheduler registry tests
##############################
sprokit_build_test(scheduler_registry libraries test_scheduler_registry.cxx)

sprokit_make_test(scheduler_registry get_twice)
sprokit_make_test(scheduler_registry null_config)
sprokit_make_test(scheduler_registry null_pipeline)
sprokit_make_test(scheduler_registry load_schedulers)
sprokit_make_test(scheduler_registry null_ctor)
sprokit_make_test(scheduler_registry duplicate_types)
sprokit_make_test(scheduler_registry unknown_types)
sprokit_make_test(scheduler_registry module_marking)

##############################
# Running tests
##############################
sprokit_build_test(run libraries test_run.cxx)

set(schedulers
  sync
  thread_per_process)

if (SPROKIT_ENABLE_PYTHON)
  set(schedulers
    ${schedulers}
    pythread_per_process)
endif ()

function (sprokit_make_run_test group instance)
  foreach (scheduler ${schedulers})
    if (SPROKIT_ENABLE_PYTHON)
      if (${scheduler} STREQUAL "pythread_per_process" OR
          ${instance} STREQUAL "pysimple_pipeline")
        set(test_environment "PYTHONPATH=${python_module_path}${cur_python_path}")
      endif ()
    endif ()

    sprokit_make_test(${group} ${instance}-${scheduler})

    set_tests_properties(test-${group}-${instance}-${scheduler}
      PROPERTIES
        TIMEOUT 5)

    unset(test_environment)
  endforeach ()
endfunction ()

sprokit_make_run_test(run simple_pipeline)
if (SPROKIT_ENABLE_PYTHON)
  sprokit_make_run_test(run pysimple_pipeline)
endif ()
sprokit_make_run_test(run multiplier_pipeline)
sprokit_make_run_test(run multiplier_cluster_pipeline)
