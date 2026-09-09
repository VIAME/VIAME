# VIAME video_io tests

if( VIAME_ENABLE_PYTHON )
  viame_add_pytest_test(
    NAME    "unit:video_io:writer"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_pyav_video_output.py"
    LABELS  UNIT
    TIMEOUT 600
    SOURCE_SETUP
  )
endif()
