project(kwiver_logger_tests)

include(test-setup)

##############################
# Core tests
##############################

kwiver_discover_tests(logger             test_libraries test_logger.cxx)
