project(kwiver_vital_metadata_tests)

include(kwiver-test-setup)

set( test_libraries vital vital_vpm vital_algo vital_video_metadata )

##############################
# Vital Video Metadata tests
##############################

kwiver_discover_tests(vital_metadata_pos_io    test_libraries test_pos_io.cxx  "${kwiver_test_data_directory}"  )
