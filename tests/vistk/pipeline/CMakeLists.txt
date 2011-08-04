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

vistk_make_test(edge makes_dependency)
vistk_make_test(edge new_has_no_data)
vistk_make_test(edge new_is_not_full)
vistk_make_test(edge new_has_count_zero)
vistk_make_test(edge push_datum)
vistk_make_test(edge peek_datum)
vistk_make_test(edge pop_datum)
vistk_make_test(edge get_datum)
vistk_make_test(edge required_by_downstream)
vistk_make_test(edge null_upstream_process)
vistk_make_test(edge null_downstream_process)
vistk_make_test(edge set_upstream_process)
vistk_make_test(edge set_downstream_process)

##############################
# Pipeline tests
##############################
vistk_build_test(pipeline libraries test_pipeline.cxx)

vistk_make_test(pipeline null_process)
vistk_make_test(pipeline add_process)
vistk_make_test(pipeline add_group)
vistk_make_test(pipeline duplicate_process_process)
vistk_make_test(pipeline duplicate_process_group)
