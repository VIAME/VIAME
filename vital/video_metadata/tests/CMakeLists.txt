project(kwiver_vital_metadata_tests)

include(kwiver-test-setup)

set(test_libraries vital vital_vpm vital_algo vital_video_metadata )

##############################
# Vital Video Metadata tests
##############################

kwiver_discover_gtests(vital metadata_pos_io LIBRARIES ${test_libraries} ARGUMENTS "${kwiver_test_data_directory}")
