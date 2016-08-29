project(arrows_test_matlab)

include(kwiver-test-setup)

set(test_libraries      vital kwiver_algo kwiver_algo_matlab kwiver_algo_ocv ${Matlab_LIBRARIES} )

##############################
# Algorithms Matlab tests
##############################
kwiver_discover_tests(matlab_util                test_libraries test_matlab_util.cxx "${kwiver_test_data_directory}" )
