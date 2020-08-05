project( track_oracle_tests )

include( kwiver-test-setup )

set( test_libraries  track_oracle data_terms track_oracle_file_formats )

##
##
##

kwiver_discover_gtests( track_oracle basic_functions LIBRARIES ${test_libraries} )
kwiver_discover_gtests( track_oracle thread_safety LIBRARIES ${test_libraries} ARGUMENTS "${kwiver_test_data_directory}" )
kwiver_discover_gtests( track_oracle kpf_geometry LIBRARIES ${test_libraries} ARGUMENTS "${kwiver_test_data_directory}" )
kwiver_discover_gtests( track_oracle kpf_activity LIBRARIES ${test_libraries} ARGUMENTS "${kwiver_test_data_directory}" )
