project(algorithms_test_ceres)

include(arrows-test-setup)

set(test_libraries algorithms_ceres)

##############################
# Algorithms Ceres tests
##############################
kwiver_discover_tests(ceres_bundle_adjust       test_libraries test_bundle_adjust.cxx)
kwiver_discover_tests(ceres_reprojection_error  test_libraries test_reprojection_error.cxx)
