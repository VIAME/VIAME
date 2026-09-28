# Tests against OpenCV
#
# The only tests in the tree that import cv2. See README.md for why they are
# here and why they are kept.
#
# Labelled UNIT as well as REFERENCE so the ordinary sweep still runs them,
# and REFERENCE so every reference test can be selected or excluded together:
#
#     ctest -L REFERENCE      every reference test, these and the replays
#     ctest -LE REFERENCE     everything else

project( test_viame_reference_opencv )

include( viame-test-setup )

if( VIAME_ENABLE_PYTHON )
  viame_add_test(
    NAME    "reference:opencv:kernels_parallel"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_kernels_parallel.py"
    LABELS  UNIT REFERENCE
    TIMEOUT 120
    SOURCE_SETUP
  )

  viame_add_test(
    NAME    "reference:opencv:kernels_streaming"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_kernels_streaming.py"
    LABELS  UNIT REFERENCE
    TIMEOUT 120
    SOURCE_SETUP
  )
endif()
