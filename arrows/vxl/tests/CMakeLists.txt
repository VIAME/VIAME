project(arrows_test_vxl)

include(kwiver-test-setup)

set(test_libraries     kwiver_algo_vxl)

##############################
# Algorithms VXL tests
##############################
kwiver_discover_gtests(vxl aligned_edge_detection      LIBRARIES ${test_libraries}  ARGUMENTS "${kwiver_test_data_directory}")
kwiver_discover_gtests(vxl bounding_box                LIBRARIES ${test_libraries})
kwiver_discover_gtests(vxl camera                      LIBRARIES ${test_libraries})
kwiver_discover_gtests(vxl estimate_homography         LIBRARIES ${test_libraries})
kwiver_discover_gtests(vxl estimate_similarity         LIBRARIES ${test_libraries})
kwiver_discover_gtests(vxl image                       LIBRARIES ${test_libraries})
kwiver_discover_gtests(vxl polygon                     LIBRARIES ${test_libraries})
kwiver_discover_gtests(vxl vidl_ffmpeg_video_input     LIBRARIES ${test_libraries}  ARGUMENTS "${kwiver_test_data_directory}")


# Additional tests that depend on the MVG arrow
list(APPEND test_libraries kwiver_algo_mvg)

kwiver_discover_gtests(vxl bundle_adjust               LIBRARIES ${test_libraries})
kwiver_discover_gtests(vxl estimate_essential_matrix   LIBRARIES ${test_libraries})
kwiver_discover_gtests(vxl estimate_fundamental_matrix LIBRARIES ${test_libraries})
kwiver_discover_gtests(vxl optimize_cameras            LIBRARIES ${test_libraries})
kwiver_discover_gtests(vxl triangulate_landmarks       LIBRARIES ${test_libraries})

# This test is really for an algorithm in MVG, but it's here
# because the test depends on availability of VXL sub-algorithms
kwiver_discover_gtests(vxl initialize_cameras_landmarks LIBRARIES ${test_libraries})
