project(kwiver_util_tests)

include(kwiver-test-setup)

set( test_libraries )

##############################
# Range tests
##############################

kwiver_discover_gtests(vital range_filter       LIBRARIES ${test_libraries})
kwiver_discover_gtests(vital range_indirect     LIBRARIES ${test_libraries})
kwiver_discover_gtests(vital range_iota         LIBRARIES ${test_libraries})
kwiver_discover_gtests(vital range_sliding      LIBRARIES ${test_libraries})
kwiver_discover_gtests(vital range_transform    LIBRARIES ${test_libraries})
kwiver_discover_gtests(vital range_valid        LIBRARIES ${test_libraries})
