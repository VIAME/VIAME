project(kwiver_logger_tests)

include(test-setup)

set( test_libraries kwiver_logger )

##############################
# Logger tests
##############################

kwiver_discover_tests(logger             test_libraries test_logger.cxx)
