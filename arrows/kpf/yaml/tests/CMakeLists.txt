project( kpf_tests )

include( kwiver-test-setup )

set( test_libraries vital_kpf vital_logger )

kwiver_discover_tests( test_kpf     test_libraries test_kpf.cxx )

