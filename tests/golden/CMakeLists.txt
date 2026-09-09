# Golden behaviour tests
#
# Each recording under tests/golden/<group>/ is the output of an
# implementation that is being replaced. The test replays every recorded case
# against whatever this build registers under the same name, so a replacement
# has to reproduce it.

if( VIAME_ENABLE_PYTHON )
  viame_add_pytest_test(
    NAME    "golden:video"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_video.py"
    LABELS  GOLDEN
    TIMEOUT 1800
    SOURCE_SETUP
  )

  viame_add_pytest_test(
    NAME    "golden:replay"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_golden.py"
    LABELS  GOLDEN
    TIMEOUT 1800
    SOURCE_SETUP
  )
endif()
