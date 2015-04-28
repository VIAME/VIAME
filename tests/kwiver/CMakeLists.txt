project(kwiver_core_tests)

##############################
# Core tests
##############################
kwiver_discover_tests(core_config             test_libraries test_config_block.cxx)
kwiver_discover_tests(core_config_io          test_libraries test_config_block_io.cxx
  "${kwiver_test_data_directory}"
  )
kwiver_discover_tests(core_homography         test_libraries test_homography.cxx)
kwiver_discover_tests(core_image              test_libraries test_image.cxx)
# kwiver_discover_tests(core_transform          test_libraries test_transform.cxx)
kwiver_discover_tests(core_vector             test_libraries test_vector.cxx)

kwiver_discover_tests(core_logger             test_libraries test_logger.cxx)
