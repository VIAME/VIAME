project(kwiver_klv_tests)

include(kwiver-test-setup)

set( test_libraries vital vital_vpm )

##############################
# KLV tests
##############################

kwiver_discover_tests(KLV             test_libraries test_klv.cxx)
