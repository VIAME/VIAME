# VIAME video_io tests

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
