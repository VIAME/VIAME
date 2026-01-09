# VIAME OpenCV Plugin Tests

project( test_viame_opencv )

include( kwiver-test-setup )

find_package( OpenCV REQUIRED )

# Include directories for plugin headers
include_directories( ${VIAME_SOURCE_DIR}/plugins/opencv )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set( test_libraries viame_opencv GTest::gtest_main )

kwiver_discover_gtests( viame_opencv filter_stereo_feature_tracks
  LIBRARIES ${test_libraries}
  PROPERTIES LABELS "VIAME;UNIT" )
# TODO: Enable when ocv_kmedians.h is implemented
# kwiver_discover_gtests( viame_opencv kmedians LIBRARIES ${test_libraries} )
