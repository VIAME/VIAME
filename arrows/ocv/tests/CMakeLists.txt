project(arrows_test_ocv)

include(kwiver-test-setup)

set(test_libraries      vital vital_vpm kwiver_algo_ocv )

##############################
# Algorithms OpenCV tests
##############################
kwiver_discover_tests(ocv_estimate_fundamental_matrix test_libraries test_estimate_fundamental_matrix.cxx)
kwiver_discover_tests(ocv_estimate_homography test_libraries test_estimate_homography.cxx)
kwiver_discover_tests(ocv_image               test_libraries test_image.cxx)
kwiver_discover_tests(ocv_feature_set         test_libraries test_feature_set.cxx)
kwiver_discover_tests(ocv_descriptor_set      test_libraries test_descriptor_set.cxx)
kwiver_discover_tests(ocv_match_set           test_libraries test_match_set.cxx)
kwiver_discover_tests(ocv_algo_config         test_libraries test_algo_config.cxx)
kwiver_discover_tests(ocv_distortion          test_libraries test_distortion.cxx)
kwiver_discover_tests(ocv_bounding_box        test_libraries test_bounding_box.cxx)
