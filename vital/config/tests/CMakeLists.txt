project(kwiver_config_tests)

include(kwiver-test-setup)

set( test_libraries vital_config kwiversys )


##############################
# Config Tests
##############################
kwiver_discover_tests(vital-config        test_libraries     test_config_block.cxx )
kwiver_discover_tests(vital-config_io     test_libraries     test_config_block_io.cxx "${kwiver_test_data_directory}" )
