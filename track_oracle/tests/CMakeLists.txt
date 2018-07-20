project( track_oracle_tests )

include( kwiver-test-setup )

set( test_libraries  track_oracle data_terms )

##
##
##

kwiver_discover_gtests( track_oracle basic_functions LIBRARIES ${test_libraries} )
