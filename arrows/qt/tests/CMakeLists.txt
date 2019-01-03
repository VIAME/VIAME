project(arrows_test_qt)

include(kwiver-test-setup)

set(test_libraries     kwiver_algo_qt)

##############################
# Algorithms Qt tests
##############################
kwiver_discover_gtests(qt image LIBRARIES ${test_libraries})
