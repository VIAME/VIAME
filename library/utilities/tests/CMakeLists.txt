# VIAME utilities tests

project( test_viame_utilities )

include( viame-test-setup )

if( VIAME_ENABLE_PYTHON )
  # The compatibility packages phase 2 left behind: every old module path
  # must name a module that exists, and import as the same module object.
  viame_add_test(
    NAME    "unit:utilities:module_aliases"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_module_aliases.py"
    LABELS  UNIT
    TIMEOUT 300
    SOURCE_SETUP
  )
endif()

# From `tests/plugins/core` in P2-T09. Linked as they were there, against the
# libraries the core plugin's tests shared, so moving them changed nothing
# but where they are.
set( moved_core_test_libraries
  viame_utilities
  viame_image_kernels
  viame_file_io
  viame_training
  viame_evaluation
  kwiver::vital
  kwiver::vital_algo
  GTest::gtest_main )

# Imported from upstream in the merge of 17c43764b.
viame_discover_gtests( viame manipulate_pipelines
  LIBRARIES ${moved_core_test_libraries} )

if( VIAME_ENABLE_PYTHON )
  viame_add_test(
    NAME    "unit:utilities:model_wrap"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_model_wrap.py"
    LABELS  UNIT CORE
    TIMEOUT 300
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
  )

  # The OpenCV replacements. `test_geometry.py` and `test_imageops.py` were
  # written with the first of them and never registered, so they had never
  # run in a tier -- 18 and 6 assertions sitting in the tree doing nothing.
  # Registered together here so that does not happen to the third.
  viame_add_test(
    NAME    "unit:utilities:geometry"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_geometry.py"
    LABELS  UNIT CORE
    TIMEOUT 300
    SOURCE_SETUP
  )

  viame_add_test(
    NAME    "unit:utilities:imageops"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_imageops.py"
    LABELS  UNIT CORE
    TIMEOUT 300
    SOURCE_SETUP
  )

  # Zhang's calibration, which replaces cv::calibrateCamera. CORE because
  # the stereo measurement chain stands on it.
  viame_add_test(
    NAME    "unit:utilities:calibration"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_calibration.py"
    LABELS  UNIT CORE
    TIMEOUT 600
    SOURCE_SETUP
  )
endif()
