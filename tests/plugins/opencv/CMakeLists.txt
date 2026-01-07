# VIAME OpenCV Plugin Tests

project( test_opencv_plugin )

include( kwiver-test-setup )

find_package( OpenCV REQUIRED )

# Include directories for plugin headers
include_directories( ${VIAME_SOURCE_DIR}/plugins/opencv )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set( test_libraries viame_opencv GTest::gtest_main )

kwiver_discover_gtests( opencv_plugin ocv_filter_stereo_feature_tracks LIBRARIES ${test_libraries} )
kwiver_discover_gtests( opencv_plugin ocv_kmedians LIBRARIES ${test_libraries} )
