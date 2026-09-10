# VIAME object_detectors tests

project( test_viame_object_detectors )

include( kwiver-test-setup )

if( VIAME_ENABLE_PYTHON )
  # `hough_circle` is python since P7-T04, and the registry dump cannot read
  # a python implementation's config, so its config contract is held here.
  viame_add_pytest_test(
    NAME    "unit:object_detectors:hough_circle"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_hough_circle_detector.py"
    LABELS  UNIT
    TIMEOUT 600
    SOURCE_SETUP
  )
endif()
