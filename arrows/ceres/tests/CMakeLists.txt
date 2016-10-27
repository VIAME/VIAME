project(arrows_test_ceres)

include(kwiver-test-setup)

set(test_libraries    kwiver_algo_ceres)
if(NOT CERES_USES_MINILOG)
  list(APPEND test_libraries glog)
endif()

##############################
# Algorithms Ceres tests
##############################
kwiver_discover_tests(ceres_bundle_adjust       test_libraries test_bundle_adjust.cxx)
kwiver_discover_tests(ceres_reprojection_error  test_libraries test_reprojection_error.cxx)
