project(arrows_adapters_tests)

include(kwiver-test-setup)

set( test_libraries vital sprokit_pipeline sprokit_tools kwiver_adapter )

#############################
# adapter process tests
#############################

kwiver_discover_tests(adapter_basic             test_libraries basic_test.cxx)
