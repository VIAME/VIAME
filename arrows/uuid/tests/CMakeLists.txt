project(arrows_test_uuid)

include(kwiver-test-setup)

set( test_libraries    kwiver_algo_uuid vital vital_vpm )

##############################
# Algorithms uuid plugin tests
##############################
kwiver_discover_gtests(uuid uuid_factory LIBRARIES ${test_libraries})
