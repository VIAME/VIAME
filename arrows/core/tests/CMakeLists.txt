project(arrows_test_core)

include(kwiver-test-setup)

set( test_libraries    kwiver_algo_core vital vital_vpm )

##############################
# Algorithms core plugin tests
##############################
kwiver_discover_gtests(core detected_object_io        LIBRARIES ${test_libraries})
kwiver_discover_gtests(core dynamic_configuration     LIBRARIES ${test_libraries})
kwiver_discover_gtests(core epipolar_geometry         LIBRARIES ${test_libraries})
kwiver_discover_gtests(core feature_descriptor_io     LIBRARIES ${test_libraries})
kwiver_discover_gtests(core interpolate_camera        LIBRARIES ${test_libraries})
kwiver_discover_gtests(core interpolate_track_spline  LIBRARIES ${test_libraries})
kwiver_discover_gtests(core track_set_impl            LIBRARIES ${test_libraries})
kwiver_discover_gtests(core triangulate_landmarks     LIBRARIES ${test_libraries})
kwiver_discover_gtests(core video_input_filter        LIBRARIES ${test_libraries} ARGUMENTS "${kwiver_test_data_directory}")
kwiver_discover_gtests(core video_input_image_list    LIBRARIES ${test_libraries} ARGUMENTS "${kwiver_test_data_directory}")
kwiver_discover_gtests(core video_input_pos           LIBRARIES ${test_libraries} ARGUMENTS "${kwiver_test_data_directory}")
kwiver_discover_gtests(core video_input_split         LIBRARIES ${test_libraries} ARGUMENTS "${kwiver_test_data_directory}")
