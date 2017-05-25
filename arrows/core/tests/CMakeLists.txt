project(arrows_test_core)

include(kwiver-test-setup)

set( test_libraries    kwiver_algo_core vital vital_vpm )

##############################
# Algorithms core plugin tests
##############################
kwiver_discover_tests(core_triangulate_landmarks        test_libraries test_triangulate_landmarks.cxx)
kwiver_discover_tests(core_epipolar_geometry            test_libraries test_epipolar_geometry.cxx)
kwiver_discover_tests(core_interpolate_camera           test_libraries test_interpolate_camera.cxx)

kwiver_discover_tests(core_detected_object_io           test_libraries test_detected_object_io.cxx)
kwiver_discover_tests(core_dynamic_configuration        test_libraries test_dynamic_config.cxx)

if( LIB_UUID )
kwiver_discover_tests(core_uuid_factory                 test_libraries test_uuid_factory.cxx)
endif()
