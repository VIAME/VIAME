project(kwiver_core_tests)

include(test-setup)

##############################
# Core tests
##############################

kwiver_discover_tests(core_homography         test_libraries test_homography.cxx)
kwiver_discover_tests(core_image              test_libraries test_image.cxx)
# kwiver_discover_tests(core_transform          test_libraries test_transform.cxx)
kwiver_discover_tests(core_vector             test_libraries test_vector.cxx)
