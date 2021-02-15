project(sprokit_adapters_tests)

set(CMAKE_FOLDER "Sprokit/Tests")

include(kwiver-test-setup)

set( test_libraries       vital sprokit_pipeline sprokit_pipeline_util kwiver_adapter )

#############################
# adapter process tests
#############################

kwiver_discover_tests(adapter_basic             test_libraries basic_test.cxx)

#FIXME: This test probably isn't in the right location
#       but keeping it here makes the deps work out correctly.
#       Previously the test was in sprokit, but required processes to be enabled.
#       That is a less intuative sequence since sprokit doesn't require processes.
kwiver_discover_tests(non_blocking test_libraries test_non_blocking.cxx)
