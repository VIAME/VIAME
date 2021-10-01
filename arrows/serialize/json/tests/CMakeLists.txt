project(kwiver_serialize_json_tests)

set(CMAKE_FOLDER "Arrows/Serialize/Tests")

include(kwiver-test-setup)

set( test_libraries vital vital_vpm vital_algo kwiver_serialize_json )

##############################
# JSON tests
##############################

kwiver_discover_gtests(serialize-json serialize         LIBRARIES ${test_libraries})
kwiver_discover_gtests(serialize-json load_save         LIBRARIES ${test_libraries})
