project(kwiver_util_tests)

include(kwiver-test-setup)

set( test_libraries vital vital_vpm kwiversys )

##############################
# Util tests
##############################

kwiver_discover_tests(util_timer                test_libraries test_timer.cxx)
kwiver_discover_tests(util_any_converter        test_libraries test_any_convert.cxx)
kwiver_discover_tests(util_string_editor        test_libraries test_string_editor.cxx)
kwiver_discover_tests(util_string               test_libraries test_string.cxx)
kwiver_discover_tests(util_data_stream_reader   test_libraries test_data_stream_reader.cxx)
kwiver_discover_tests(util_thread_pool          test_libraries test_thread_pool.cxx)
kwiver_discover_tests(util_token_expander       test_libraries test_token_expander.cxx)
