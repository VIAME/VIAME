project(plugin_loader_tests)

set(CMAKE_FOLDER "Vital/Tests")

include(kwiver-test-setup)

set( test_libraries vital vital_vpm )

##############################
# Loader tests
##############################

kwiver_discover_gtests(vital plugin_loader LIBRARIES ${test_libraries})
