project(vistk_test_lua_pipeline_util)

set(vistk_test_pipelines_directory
  ${vistk_test_data_directory}/pipelines)

function (vistk_make_lua_load_test name instance)
  vistk_make_lua_test(${name} ${instance}
    "${vistk_test_pipelines_directory}")
endfunction (vistk_make_lua_load_test name instance)
