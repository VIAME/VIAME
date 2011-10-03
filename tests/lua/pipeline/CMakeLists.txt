project(vistk_test_lua_pipeline)

##############################
# Config tests
##############################
vistk_build_lua_test(config test-config.lua)

vistk_make_lua_test(config import)
vistk_make_lua_test(config create)
vistk_make_lua_test(config has_value)
vistk_make_lua_test(config get_value)
vistk_make_lua_test(config get_value_no_exist)
vistk_make_lua_test(config unset_value)
vistk_make_lua_test(config available_values)
vistk_make_lua_test(config read_only)
vistk_make_lua_test(config read_only_unset)
vistk_make_lua_test(config subblock)
vistk_make_lua_test(config subblock_view)
vistk_make_lua_test(config merge_config)

##############################
# Datum tests
##############################
vistk_build_lua_test(datum test-datum.lua)

vistk_make_lua_test(datum import)
vistk_make_lua_test(datum new)
vistk_make_lua_test(datum empty)
vistk_make_lua_test(datum complete)
vistk_make_lua_test(datum error)

##############################
# Edge tests
##############################
vistk_build_lua_test(edge test-edge.lua)

vistk_make_lua_test(edge import)
vistk_make_lua_test(edge create)
vistk_make_lua_test(edge datum_create)
vistk_make_lua_test(edge api_calls)
vistk_make_lua_test(edge datum_api_calls)

##############################
# Modules tests
##############################
vistk_build_lua_test(modules test-modules.lua)

vistk_make_lua_test(modules import)
vistk_make_lua_test(modules load)

##############################
# Pipeline tests
##############################
vistk_build_lua_test(pipeline test-pipeline.lua)

vistk_make_lua_test(pipeline import)
vistk_make_lua_test(pipeline create)
vistk_make_lua_test(pipeline api_calls)

##############################
# Process tests
##############################
vistk_build_lua_test(process test-process.lua)

vistk_make_lua_test(process import)
vistk_make_lua_test(process create)
vistk_make_lua_test(process api_calls)

##############################
# Process registry tests
##############################
vistk_build_lua_test(process_registry test-process_registry.lua)

vistk_make_lua_test(process_registry import)
vistk_make_lua_test(process_registry create)
vistk_make_lua_test(process_registry api_calls)
vistk_make_lua_test(process_registry register)
vistk_make_lua_test(process_registry wrapper_api)

##############################
# Schedule tests
##############################
vistk_build_lua_test(schedule test-schedule.lua)

vistk_make_lua_test(schedule import)

##############################
# Schedule registry tests
##############################
vistk_build_lua_test(schedule_registry test-schedule_registry.lua)

vistk_make_lua_test(schedule_registry import)
vistk_make_lua_test(schedule_registry create)
vistk_make_lua_test(schedule_registry api_calls)
vistk_make_lua_test(schedule_registry register)

##############################
# Stamp tests
##############################
vistk_build_lua_test(stamp test-stamp.lua)

vistk_make_lua_test(stamp import)
vistk_make_lua_test(stamp create)
vistk_make_lua_test(stamp api_calls)

##############################
# Utils tests
##############################
vistk_build_lua_test(utils test-utils.lua)

vistk_make_lua_test(utils import)
vistk_make_lua_test(utils create)
vistk_make_lua_test(utils name_thread)
