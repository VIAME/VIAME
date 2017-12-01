project(kwiver_logger_tests)

include(kwiver-test-setup)

set( test_libraries vital_logger )

##############################
# Logger tests
##############################

kwiver_discover_gtests(vital logger LIBRARIES ${test_libraries})
