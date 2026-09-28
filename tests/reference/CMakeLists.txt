# Reference tests
#
# A reference test holds a VIAME implementation to something outside itself:
# a recording of what the code it replaced produced, or a live call into the
# library it replaced. README.md says which is which.
#
# `golden:*` keep their names and the GOLDEN label, which the documented
# verification commands select on. REFERENCE is added beside it so that every
# reference test -- these and the live ones under `opencv/` -- can be taken
# together:
#
#     ctest -L REFERENCE      every reference test
#     ctest -LE REFERENCE     everything else

project( test_viame_reference )

include( viame-test-setup )

# The live comparisons, which need the reference package installed.
add_subdirectory( opencv )

# `eigen/` holds a recording and its recorder, no tests: it is read by
# `tests/library/core_types/test_math.cxx`, registered there.

if( VIAME_ENABLE_PYTHON )
  viame_add_test(
    NAME    "golden:video"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_video.py"
    LABELS  GOLDEN REFERENCE
    TIMEOUT 1800
    SOURCE_SETUP
  )

  # Not a recording: video through a whole pipeline, which is the gap that
  # let two pipeline-breaking faults past `golden:video`.
  viame_add_test(
    NAME    "pipeline:video"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_video_pipeline.py"
    LABELS  GOLDEN REFERENCE
    TIMEOUT 900
    SOURCE_SETUP
  )

  viame_add_test(
    NAME    "golden:replay"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_golden.py"
    LABELS  GOLDEN REFERENCE
    TIMEOUT 1800
    SOURCE_SETUP
  )
endif()
