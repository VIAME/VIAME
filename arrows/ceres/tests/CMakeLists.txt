project(arrows_test_ceres)

include(kwiver-test-setup)

set(test_libraries    kwiver_algo_ceres)

# we need to link tests to GLog becuase they instantiate Ceres templates which
# use GLog.  If Ceres was built with MiniGlog, link to ceres instead to get
# MiniGlog symbols.
if(CERES_USES_MINIGLOG)
  list(APPEND test_libraries ceres)
else()
  list(APPEND test_libraries glog)
endif()

##############################
# Algorithms Ceres tests
##############################
kwiver_discover_tests(ceres_bundle_adjust       test_libraries test_bundle_adjust.cxx)
kwiver_discover_tests(ceres_optimize_cameras    test_libraries test_optimize_cameras.cxx)
kwiver_discover_tests(ceres_reprojection_error  test_libraries test_reprojection_error.cxx)
