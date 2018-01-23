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
kwiver_discover_gtests(ceres bundle_adjust      LIBRARIES ${test_libraries})
kwiver_discover_gtests(ceres optimize_cameras   LIBRARIES ${test_libraries})
kwiver_discover_gtests(ceres reprojection_error LIBRARIES ${test_libraries})
