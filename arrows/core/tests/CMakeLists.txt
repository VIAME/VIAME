project(arrows_test_core)

include(kwiver-test-setup)

set( test_libraries    kwiver_algo_core vital vital_vpm vital_video_metadata)

##############################
# Algorithms core plugin tests
##############################
kwiver_discover_tests(core_triangulate_landmarks        test_libraries test_triangulate_landmarks.cxx)
kwiver_discover_tests(core_epipolar_geometry            test_libraries test_epipolar_geometry.cxx)
kwiver_discover_tests(core_feature_descriptor_io        test_libraries test_feature_descriptor_io.cxx)
kwiver_discover_tests(core_interpolate_camera           test_libraries test_interpolate_camera.cxx)
kwiver_discover_tests(core_detected_object_io           test_libraries test_detected_object_io.cxx)
kwiver_discover_tests(core_dynamic_configuration        test_libraries test_dynamic_config.cxx)
kwiver_discover_tests(core_video_input_image_list       test_libraries test_video_input_image_list.cxx  "${kwiver_test_data_directory}" )
kwiver_discover_tests(core_video_input_pos              test_libraries test_video_input_pos.cxx  "${kwiver_test_data_directory}" )
kwiver_discover_tests(core_video_input_split            test_libraries test_video_input_split.cxx  "${kwiver_test_data_directory}" )
kwiver_discover_tests(core_video_input_filter           test_libraries test_video_input_filter.cxx  "${kwiver_test_data_directory}" )
