project(arrows_test_mvg)

set(CMAKE_FOLDER "Arrows/MVG/Tests")

include(kwiver-test-setup)

set( test_libraries    kwiver_algo_mvg vital vital_vpm )

##############################
# Algorithms MVG plugin tests
##############################
kwiver_discover_gtests(mvg epipolar_geometry         LIBRARIES ${test_libraries})
kwiver_discover_gtests(mvg integrate_depth_maps      LIBRARIES ${test_libraries} kwiver_algo_core)
kwiver_discover_gtests(mvg interpolate_camera        LIBRARIES ${test_libraries})
kwiver_discover_gtests(mvg triangulate_landmarks     LIBRARIES ${test_libraries})
kwiver_discover_gtests(mvg triangulate_landmarks_rpc LIBRARIES ${test_libraries} ARGUMENTS "${kwiver_test_data_directory}")
