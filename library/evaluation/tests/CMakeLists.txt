# VIAME evaluation tests

project( test_viame_evaluation )

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

viame_discover_gtests( viame evaluate_models
  LIBRARIES ${moved_core_test_libraries} )

# What `viame score --output-plots` writes beside its images, recorded before
# P7-T07 handed the drawing to matplotlib. The pictures change; these do not.
viame_discover_gtests( viame plot_data_export
  LIBRARIES ${moved_core_test_libraries} )

target_compile_definitions( test-viame-plot_data_export PRIVATE
  "VIAME_PLOT_DATA_EXPORT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"" )
