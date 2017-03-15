project(plugin_loader_tests)

include(kwiver-test-setup)

set( test_libraries vital vital_vpm )

##############################
# Loader tests
##############################

kwiver_discover_tests(vital-plugin-loader             test_libraries test_plugin_loader.cxx)
