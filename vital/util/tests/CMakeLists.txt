project(kwiver_util_tests)

include(kwiver-test-setup)

set( test_libraries vital vital_vpm kwiversys )

##############################
# Util tests
##############################

kwiver_discover_gtests(vital any_converter      LIBRARIES ${test_libraries})
kwiver_discover_gtests(vital data_stream_reader LIBRARIES ${test_libraries})
kwiver_discover_gtests(vital string             LIBRARIES ${test_libraries})
kwiver_discover_gtests(vital string_editor      LIBRARIES ${test_libraries})
kwiver_discover_gtests(vital thread_pool        LIBRARIES ${test_libraries})
kwiver_discover_gtests(vital timer              LIBRARIES ${test_libraries})
kwiver_discover_gtests(vital token_expander     LIBRARIES ${test_libraries})
