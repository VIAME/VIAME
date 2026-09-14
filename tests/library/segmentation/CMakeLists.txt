# VIAME segmentation tests

project( test_viame_segmentation )

include( viame-test-setup )

if( VIAME_ENABLE_PYTHON )
  # `python -m viame.core.interactive_service`, the command DIVE's desktop
  # client starts, forwarded to `viame.segmentation.interactive_service`
  # since P2-T07 emptied `viame.core`.
  viame_add_pytest_test(
    NAME    "unit:segmentation:dive_service_path"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_dive_service_path.py"
    LABELS  UNIT
    TIMEOUT 300
    SOURCE_SETUP
  )
endif()
