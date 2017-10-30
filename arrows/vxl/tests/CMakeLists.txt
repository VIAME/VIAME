project(arrows_test_vxl)

include(kwiver-test-setup)

set(test_libraries     kwiver_algo_vxl)

##############################
# Algorithms VXL tests
##############################
kwiver_discover_gtests(vxl bounding_box                LIBRARIES ${test_libraries})
kwiver_discover_gtests(vxl bundle_adjust               LIBRARIES ${test_libraries})
kwiver_discover_gtests(vxl camera                      LIBRARIES ${test_libraries})
kwiver_discover_tests(vxl_estimate_essential_matrix    test_libraries test_estimate_essential_matrix.cxx)
kwiver_discover_tests(vxl_estimate_fundamental_matrix  test_libraries test_estimate_fundamental_matrix.cxx)
kwiver_discover_tests(vxl_estimate_homography          test_libraries test_estimate_homography.cxx)
kwiver_discover_tests(vxl_estimate_similarity          test_libraries test_estimate_similarity.cxx)
kwiver_discover_gtests(vxl image                       LIBRARIES ${test_libraries})
kwiver_discover_tests(vxl_optimize_cameras             test_libraries test_optimize_cameras.cxx)
kwiver_discover_gtests(vxl polygon                     LIBRARIES ${test_libraries})
kwiver_discover_gtests(vxl triangulate_landmarks       LIBRARIES ${test_libraries})

# This test is really for an algorithm in core, but it's here
# because the test depends on availablity of VXL sub-algorithms
set(test_libraries     kwiver_algo_vxl kwiver_algo_core)

kwiver_discover_tests(vxl_initialize_cameras_landmarks test_libraries test_initialize_cameras_landmarks.cxx)
