project(arrows_test_database)

include(kwiver-test-setup)

set(test_libraries kwiver_algo_database)

##############################
# Algorithms VXL tests
##############################
kwiver_discover_tests(kwiver_database test_libraries test_kwiver_database.cxx)
