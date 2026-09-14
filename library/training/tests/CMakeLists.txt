# VIAME training tests

project( test_viame_training )

include( viame-test-setup )

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

viame_discover_gtests( viame utilities_training
  LIBRARIES ${moved_core_test_libraries} )

viame_discover_gtests( viame label_files
  LIBRARIES ${moved_core_test_libraries} )
