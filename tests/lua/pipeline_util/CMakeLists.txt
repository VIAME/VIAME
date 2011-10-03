project(vistk_test_lua_pipeline_util)

set(vistk_test_pipelines_directory
  ${vistk_test_data_directory}/pipelines)

function (vistk_make_lua_load_test name instance)
  vistk_make_lua_test(${name} ${instance}
    "${vistk_test_pipelines_directory}")
endfunction (vistk_make_lua_load_test name instance)

##############################
# Bake tests
##############################
vistk_build_lua_test(bake test-bake.lua)

vistk_make_lua_load_test(bake import)
vistk_make_lua_load_test(bake simple_pipeline)

##############################
# Export tests
##############################
vistk_build_lua_test(export test-export.lua)

vistk_make_lua_load_test(export import)
vistk_make_lua_load_test(export simple_pipeline)

##############################
# Load tests
##############################
vistk_build_lua_test(load test-load.lua)

vistk_make_lua_load_test(load import)
vistk_make_lua_load_test(load create)
vistk_make_lua_load_test(load api_calls)
vistk_make_lua_load_test(load simple_pipeline)
