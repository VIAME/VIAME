# VIAME OpenCV Plugin Tests

project( test_viame_opencv )

include( kwiver-test-setup )

find_package( OpenCV REQUIRED )

# Include directories for plugin headers
include_directories( ${VIAME_SOURCE_DIR}/plugins/opencv )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set( test_libraries viame_opencv GTest::gtest_main )

# `filter_stereo_feature_tracks` and `kmedians` moved to python in P7-T06;
# their tests are `tests/library/measurement/test_stereo_frame_selection.py`
# now, and they say rather more than the three smoke tests that were here.
