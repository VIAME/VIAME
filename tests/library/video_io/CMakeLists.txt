# VIAME video_io tests

project( test_viame_video_io )

include( viame-test-setup )

if( VIAME_ENABLE_PYTHON )
  viame_add_test(
    NAME    "unit:video_io:writer"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_pyav_video_output.py"
    LABELS  UNIT
    TIMEOUT 600
    SOURCE_SETUP
  )

  viame_add_test(
    NAME    "unit:video_io:reader"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_pyav_video_input.py"
    LABELS  UNIT
    TIMEOUT 600
    SOURCE_SETUP
  )

  viame_add_test(
    NAME    "unit:video_io:frames"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_frames.py"
    LABELS  UNIT
    TIMEOUT 300
    SOURCE_SETUP
  )

  viame_add_test(
    NAME    "unit:video_io:cli_reader"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_ffmpeg_cli_video_input.py"
    LABELS  UNIT
    TIMEOUT 600
    SOURCE_SETUP
  )
endif()

if( VIAME_ENABLE_PYTHON )
  viame_add_test( NAME "unit:video_io:vidl_color"
    TARGET "${CMAKE_CURRENT_SOURCE_DIR}/test_vidl_color.py"
    LABELS UNIT TIMEOUT 120 SOURCE_SETUP )
endif()
