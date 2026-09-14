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
  viame_add_test(
    NAME    "unit:file_io:opencv_yaml"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_opencv_yaml.py"
    LABELS  UNIT
    TIMEOUT 600
    SOURCE_SETUP
  )
endif()

# From `tests/plugins/core` in P2-T09. Linked as they were there, against the
# libraries the core plugin's tests shared, so moving them changed nothing
# but where they are.
set( moved_core_test_libraries
  viame_utilities
  viame_image_ops
  viame_file_io
  viame_training
  viame_evaluation
  kwiver::vital
  kwiver::vital_algo
  GTest::gtest_main )

viame_discover_gtests( viame viame_csv_tot_once
  LIBRARIES ${moved_core_test_libraries} )

# The stereo rig JSON, recorded before P8-T06 replaces cereal. The reader has
# a golden case as well; this holds the writer, which nothing else does.
viame_discover_gtests( viame camera_rig_json
  LIBRARIES ${moved_core_test_libraries} )

# The CSV writers' mask-to-polygon conversion, recorded before P7-T07 ported
# it off `cv::findContours` and `cv::approxPolyDP`.
viame_discover_gtests( viame mask_polygon_csv
  LIBRARIES ${moved_core_test_libraries} )

target_compile_definitions( test-viame-mask_polygon_csv PRIVATE
  "VIAME_MASK_POLYGON_CSV_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"" )

if( VIAME_ENABLE_PYTHON )
  viame_add_test(
    NAME    "unit:file_io:coco_writers"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_coco_writers.py"
    LABELS  UNIT CORE
    TIMEOUT 300
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
  )
endif()
