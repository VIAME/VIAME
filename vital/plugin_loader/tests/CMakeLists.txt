project(plugin_loader_tests)

include(kwiver-test-setup)

set( test_libraries vital vital_vpm )

##############################
# Loader tests
##############################

kwiver_discover_gtests(vital plugin_loader LIBRARIES ${test_libraries})
