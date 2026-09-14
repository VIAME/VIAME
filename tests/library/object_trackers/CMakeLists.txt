# VIAME object tracker tests

project( test_viame_object_trackers )

include( viame-test-setup )

if( VIAME_ENABLE_PYTHON )
  # The Kalman filter the ByteTrack-family trackers share since P2-T06, held
  # to a recording of the four copies it replaced.
  viame_add_pytest_test(
    NAME    "unit:object_trackers:kalman"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_kalman.py"
    LABELS  UNIT
    TIMEOUT 300
    SOURCE_SETUP
  )
endif()
