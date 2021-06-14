project(flow_processes_tests)

set(CMAKE_FOLDER "Sprokit/Tests")

include(kwiver-test-setup)

set( test_libraries       vital sprokit_pipeline sprokit_pipeline_util kwiver_adapter )

#############################
# adapter process tests
#############################

kwiver_discover_tests(mux_test             test_libraries test_mux.cxx)
