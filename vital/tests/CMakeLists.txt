project(kwiver_core_tests)

include(vital-test-setup)

set( test_libraries vital vital_apm )

##############################
# Core tests
##############################

kwiver_discover_tests(core_camera             test_libraries test_camera.cxx)
kwiver_discover_tests(core_camera_io          test_libraries test_camera_io.cxx  "${kwiver_test_data_directory}"  )
kwiver_discover_tests(core_camera_intrinsics  test_libraries test_camera_intrinsics.cxx)
kwiver_discover_tests(core_config             test_libraries test_config.cxx )
kwiver_discover_tests(core_enumerate_matrix   test_libraries test_enumerate_matrix.cxx )
kwiver_discover_tests(core_essential_matrix   test_libraries test_essential_matrix.cxx )
kwiver_discover_tests(core_fundamental_matrix test_libraries test_fundamental_matrix.cxx )
kwiver_discover_tests(core_homography         test_libraries test_homography.cxx)
kwiver_discover_tests(core_image              test_libraries test_image.cxx)
kwiver_discover_tests(core_rotation           test_libraries test_rotation.cxx)
kwiver_discover_tests(core_similarity         test_libraries test_similarity.cxx)
kwiver_discover_tests(core_track              test_libraries test_track.cxx)
kwiver_discover_tests(core_track_set          test_libraries test_track_set.cxx)
kwiver_discover_tests(core_vector             test_libraries test_vector.cxx)
## kwiver_discover_tests(core_algo               test_libraries test_algo.cxx)

kwiver_discover_tests(core_est                test_libraries test_estimate_similarity_transform.cxx)
kwiver_discover_tests(core_timestamp          test_libraries test_timestamp.cxx)
