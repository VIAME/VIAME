project(vistk_test_pipeline)

set(libraries
  vistk_pipeline)

##############################
# Introspection tests
##############################
vistk_build_test(process_introspection libraries test_process_introspection.cxx)

vistk_make_test(process_introspection all)

vistk_build_test(schedule_introspection libraries test_schedule_introspection.cxx)

vistk_make_test(schedule_introspection all)

##############################
# Config tests
##############################
vistk_build_test(config libraries test_config.cxx)

vistk_make_test(config has_value)
vistk_make_test(config get_value)
vistk_make_test(config get_value_no_exist)
vistk_make_test(config get_value_type_mismatch)
vistk_make_test(config unset_value)
vistk_make_test(config available_values)
vistk_make_test(config read_only)
