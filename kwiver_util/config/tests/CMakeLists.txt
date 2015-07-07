project(kwiver_config_tests)

include(test-setup)

set( test_libraries kwiver_config ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} )


##############################
# Config Tests
##############################
kwiver_discover_tests(config        test_libraries     test_config_block.cxx )
kwiver_discover_tests(config_io     test_libraries     test_config_block_io.cxx "${kwiver_test_data_directory}" )
