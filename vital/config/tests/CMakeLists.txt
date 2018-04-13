project(kwiver_config_tests)

include(kwiver-test-setup)

set( test_libraries vital_config vital_exceptions kwiversys )


##############################
# Config Tests
##############################
kwiver_discover_gtests(vital config_block    LIBRARIES ${test_libraries})
kwiver_discover_gtests(vital config_block_io LIBRARIES ${test_libraries} ARGUMENTS "${kwiver_test_data_directory}")
