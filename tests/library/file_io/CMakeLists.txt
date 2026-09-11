# VIAME file_io tests

project( test_viame_file_io )

include( kwiver-test-setup )

if( VIAME_ENABLE_PYTHON )
  # The OpenCV FileStorage subset, which replaced cv::FileStorage in P7-T05.
  # Driven from python because that is where the bindings are; the code under
  # test is C++.
  viame_add_pytest_test(
    NAME    "unit:file_io:opencv_yaml"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_opencv_yaml.py"
    LABELS  UNIT
    TIMEOUT 600
    SOURCE_SETUP
  )
endif()
