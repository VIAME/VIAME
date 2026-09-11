# VIAME measurement tests

project( test_viame_measurement )

include( kwiver-test-setup )

# Triangulation and the camera matrices under it. See the file's own note:
# all of this was returning zero until finding 1.20.
kwiver_discover_gtests( viame triangulate
  LIBRARIES viame_measurement
            kwiver::vital
            GTest::gtest )

# The calib3d replacement, against a recording of what OpenCV answered for
# the same rigs. Regenerate with
# `python tests/golden/projection/record_from_opencv.py`.
kwiver_discover_gtests( viame projection
  LIBRARIES viame_measurement
            kwiver::vital
            GTest::gtest )

target_compile_definitions( test-viame-projection PRIVATE
  "VIAME_GOLDEN_PROJECTION_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/../../golden/projection\"" )

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
