project(vistk_test_pipeline_util)

set(libraries
  vistk_pipeline_util
  vistk_pipeline)

set(vistk_test_pipelines_directory
  "${vistk_test_data_directory}/pipelines")

function (vistk_make_load_test name instance)
  vistk_make_test(${name} ${instance}
    "${vistk_test_pipelines_directory}")
endfunction ()

##############################
# Pipeline loading tests
##############################
set(load_pipe_libraries
  ${libraries}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
vistk_build_test(load_pipe load_pipe_libraries test_load_pipe.cxx)

vistk_make_load_test(load_pipe empty)
vistk_make_load_test(load_pipe comments)
vistk_make_load_test(load_pipe empty_config)
vistk_make_load_test(load_pipe config_block)
vistk_make_load_test(load_pipe config_block_notalnum)
vistk_make_load_test(load_pipe config_value_spaces)
vistk_make_load_test(load_pipe one_process)
vistk_make_load_test(load_pipe connected_processes)
vistk_make_load_test(load_pipe connected_processes_notalnum)
vistk_make_load_test(load_pipe config_overrides)
vistk_make_load_test(load_pipe config_read_only)
vistk_make_load_test(load_pipe config_not_a_flag)
vistk_make_load_test(load_pipe config_read_only_override)
vistk_make_load_test(load_pipe config_append)
vistk_make_load_test(load_pipe config_cappend)
vistk_make_load_test(load_pipe config_cappend_empty)
vistk_make_load_test(load_pipe config_provider_conf)
vistk_make_load_test(load_pipe config_provider_conf_dep)
vistk_make_load_test(load_pipe config_provider_conf_circular_dep)
vistk_make_load_test(load_pipe config_provider_env)
vistk_make_load_test(load_pipe config_provider_read_only)
vistk_make_load_test(load_pipe config_provider_read_only_override)
vistk_make_load_test(load_pipe config_provider_unprovided)
vistk_make_load_test(load_pipe include)
vistk_make_load_test(load_pipe no_exist)
vistk_make_load_test(load_pipe not_a_file)
vistk_make_load_test(load_pipe include_no_exist)
vistk_make_load_test(load_pipe include_not_a_file)
vistk_make_load_test(load_pipe no_parse)
vistk_make_load_test(load_pipe parse_error)
set(test_environment "VISTK_PIPE_INCLUDE_PATH=${CMAKE_CURRENT_SOURCE_DIR}")
vistk_make_load_test(load_pipe envvar)
unset(test_environment)
vistk_make_load_test(load_pipe cluster_declare)
vistk_make_load_test(load_pipe cluster_config)
vistk_make_load_test(load_pipe cluster_input_map)
vistk_make_load_test(load_pipe cluster_output_map)
vistk_make_load_test(load_pipe cluster_mappings)
vistk_make_load_test(load_pipe cluster_all)

##############################
# Export tests
##############################
set(export_dot_libraries
  ${libraries}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
vistk_build_test(export_dot export_dot_libraries test_export_dot.cxx)

vistk_make_load_test(export_dot pipeline_null)
vistk_make_load_test(export_dot simple_pipeline)
vistk_make_load_test(export_dot simple_pipeline_setup)
