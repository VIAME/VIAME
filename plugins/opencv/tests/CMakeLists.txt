project(test_opencv_plugin)

include(kwiver-test-setup)

set(test_libraries viame_opencv GTest::gtest_main)

kwiver_discover_gtests(opencv_plugin ocv_stereo_feature_track_filter LIBRARIES ${test_libraries})
kwiver_discover_gtests(opencv_plugin ocv_kmedians LIBRARIES ${test_libraries})
