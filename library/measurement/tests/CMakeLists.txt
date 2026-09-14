# VIAME measurement tests

project( test_viame_measurement )

include( viame-test-setup )

# Triangulation and the camera matrices under it. See the file's own note:
# all of this was returning zero until finding 1.20.
viame_discover_gtests( viame triangulate
  LIBRARIES viame_measurement
            kwiver::vital
            GTest::gtest )

# The calib3d replacement, against a recording of what OpenCV answered for
# the same rigs. Regenerate with
# `python tests/golden/projection/record_from_opencv.py`.
viame_discover_gtests( viame projection
  LIBRARIES viame_measurement
            kwiver::vital
            GTest::gtest )

target_compile_definitions( test-viame-projection PRIVATE
  "VIAME_GOLDEN_PROJECTION_DIR=\"${VIAME_SOURCE_DIR}/tests/golden/projection\"" )

# The stereo measurement utilities and core's detection pairing, from
# `tests/plugins/core` in P2-T06 with the code they test -- the move P2-T06
# names by file. `gtest_main` rather than `gtest`: neither file defines a
# `main` of its own.
viame_discover_gtests( viame measurement_utilities
  LIBRARIES viame_measurement
            kwiver::vital
            kwiver::vital_algo
            GTest::gtest_main )

viame_discover_gtests( viame pair_stereo_detections
  LIBRARIES viame_measurement
            kwiver::vital
            kwiver::vital_algo
            GTest::gtest_main )

if( VIAME_ENABLE_PYTHON )
  # `filter_stereo_feature_tracks` and `kmedians` are python since P7-T06.
  viame_add_test(
    NAME    "unit:measurement:frame_selection"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_stereo_frame_selection.py"
    LABELS  UNIT
    TIMEOUT 600
    SOURCE_SETUP
  )
endif()

# From `tests/plugins/core` in P2-T09: curved measurement and the interactive
# stereo points, run together as they were. `test_interactive_stereo.py` and
# `test_tracks_pairing_from_stereo.cxx` came too, unregistered as they were.
if( VIAME_ENABLE_PYTHON AND VIAME_ENABLE_OPENCV )
  viame_add_test(
    NAME    "unit:measurement:curved_measurement"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_curved_measurement.py" "${CMAKE_CURRENT_SOURCE_DIR}/test_interactive_stereo_points.py"
    LABELS  UNIT CORE
    TIMEOUT 300
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
  )
endif()

# TODO: Enable when tracks_pairing_from_stereo.h is implemented
# viame_discover_gtests( viame tracks_pairing_from_stereo LIBRARIES ... )
