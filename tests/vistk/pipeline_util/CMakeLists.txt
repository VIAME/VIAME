project(vistk_test_pipeline_util)

set(libraries
  vistk_pipeline_util
  vistk_pipeline)

set(vistk_test_pipelines_directory
  ${vistk_test_data_directory}/pipelines)

function (vistk_make_load_test name instance)
  vistk_make_test(${name} ${instance}
    "${vistk_test_pipelines_directory}")
endfunction (vistk_make_load_test name instance)

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
vistk_make_load_test(load_pipe one_process)
vistk_make_load_test(load_pipe include)
vistk_make_load_test(load_pipe no_exist)
vistk_make_load_test(load_pipe include_no_exist)
