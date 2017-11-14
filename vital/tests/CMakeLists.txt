project(kwiver_core_tests)

include(kwiver-test-setup)

set( test_libraries vital vital_vpm vital_algo )

##############################
# Core tests
##############################

kwiver_discover_tests(vital_camera             test_libraries test_camera.cxx)
kwiver_discover_tests(vital_camera_intrinsics  test_libraries test_camera_intrinsics.cxx)
kwiver_discover_tests(vital_config             test_libraries test_config.cxx )
kwiver_discover_tests(vital_enumerate_matrix   test_libraries test_enumerate_matrix.cxx )
kwiver_discover_tests(vital_essential_matrix   test_libraries test_essential_matrix.cxx )
kwiver_discover_tests(vital_fundamental_matrix test_libraries test_fundamental_matrix.cxx )
kwiver_discover_tests(vital_homography         test_libraries test_homography.cxx)
kwiver_discover_tests(vital_image              test_libraries test_image.cxx)
kwiver_discover_tests(vital_rotation           test_libraries test_rotation.cxx)
kwiver_discover_tests(vital_similarity         test_libraries test_similarity.cxx)
kwiver_discover_tests(vital_track              test_libraries test_track.cxx)
kwiver_discover_tests(vital_vector             test_libraries test_vector.cxx)
kwiver_discover_tests(vital_bounding_box       test_libraries test_bounding_box.cxx)
kwiver_discover_tests(vital_vital_uuid         test_libraries test_uuid.cxx)
kwiver_discover_tests(vital_attribute_set      test_libraries test_attribute_set.cxx)
kwiver_discover_tests(vital_polygon            test_libraries test_polygon.cxx)

## kwiver_discover_tests(vital_algo               test_libraries test_algo.cxx)

kwiver_discover_tests(vital_est                test_libraries test_estimate_similarity_transform.cxx)
kwiver_discover_tests(vital_timestamp          test_libraries test_timestamp.cxx)
kwiver_discover_tests(vital_any                test_libraries test_any.cxx)
kwiver_discover_tests(vital_algorithm_capabilities  test_libraries test_algorithm_capabilities.cxx)
kwiver_discover_tests(vital_detected_object_type     test_libraries    test_detected_object_type.cxx)
kwiver_discover_tests(vital_detected_object          test_libraries    test_detected_object.cxx)
kwiver_discover_tests(vital_detected_object_set      test_libraries    test_detected_object_set.cxx)

kwiver_discover_gtests(vital  camera_io   LIBRARIES ${test_libraries} ARGUMENTS "${kwiver_test_data_directory}" )
kwiver_discover_gtests(vital  geodesy     LIBRARIES ${test_libraries})
kwiver_discover_gtests(vital  geo_point   LIBRARIES ${test_libraries})
kwiver_discover_gtests(vital  geo_polygon LIBRARIES ${test_libraries})
kwiver_discover_gtests(vital  track_set   LIBRARIES ${test_libraries})
