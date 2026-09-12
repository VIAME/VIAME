# VIAME file_io tests

project( test_viame_file_io )

include( viame-test-setup )

###
# The KWFD feature/descriptor file
#
# Recorded before P8-T06 replaces cereal, which is what writes it. Nothing in
# this tree reads a KWFD file it did not write, so the bytes are the only
# statement of the format there is.
##
viame_discover_gtests( viame feature_descriptor_io
  LIBRARIES kwiver::vital
            kwiver::vital_algo
            kwiver::vital_config
            kwiver::vital_exceptions
            viame_file_io
            GTest::gtest )

target_include_directories( test-viame-feature_descriptor_io PRIVATE
  ${CMAKE_SOURCE_DIR}/library/file_io )

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
