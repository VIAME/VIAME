# VIAME video_io tests

project( test_viame_video_io )

include( kwiver-test-setup )

# The in-house codecs: what the formats say, on images small enough to reason
# about by hand. Agreement with OpenCV on real files is tests/golden/codecs.
kwiver_discover_gtests( viame codecs
  LIBRARIES viame_video_io
            kwiver::vital
            GTest::gtest )

if( VIAME_ENABLE_PYTHON )
  viame_add_pytest_test(
    NAME    "unit:video_io:writer"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_pyav_video_output.py"
    LABELS  UNIT
    TIMEOUT 600
    SOURCE_SETUP
  )

  viame_add_pytest_test(
    NAME    "unit:video_io:reader"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_pyav_video_input.py"
    LABELS  UNIT
    TIMEOUT 600
    SOURCE_SETUP
  )

  viame_add_pytest_test(
    NAME    "unit:video_io:cli_reader"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_ffmpeg_cli_video_input.py"
    LABELS  UNIT
    TIMEOUT 600
    SOURCE_SETUP
  )
endif()
