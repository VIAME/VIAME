# VIAME measurement tests

project( test_viame_measurement )

include( kwiver-test-setup )

# Triangulation and the camera matrices under it. See the file's own note:
# all of this was returning zero until finding 1.20.
kwiver_discover_gtests( viame triangulate
  LIBRARIES viame_measurement
            kwiver::vital
            GTest::gtest )

if( VIAME_ENABLE_PYTHON )
  # `filter_stereo_feature_tracks` and `kmedians` are python since P7-T06.
  viame_add_pytest_test(
    NAME    "unit:measurement:frame_selection"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_stereo_frame_selection.py"
    LABELS  UNIT
    TIMEOUT 600
    SOURCE_SETUP
  )
endif()
