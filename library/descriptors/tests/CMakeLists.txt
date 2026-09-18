# VIAME descriptor tests

project( test_viame_descriptors )

include( viame-test-setup )

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

# The IQR AdaBoost session. The assertions hold for any implementation of
# the interface; the recording beside them is the numbers, and its diff is
# where P7-T09's move from `cv::ml::Boost` to scikit-learn showed itself.
viame_discover_gtests( viame iqr_session_adaboost
  LIBRARIES ${moved_core_test_libraries} viame_iqr )

# The test brings the interpreter up itself; see the note on
# `python_environment`. It defines its own `main` for that, so the gtest
# main is not wanted.
if( VIAME_ENABLE_PYTHON )
  target_link_libraries( test-viame-iqr_session_adaboost
    PRIVATE ${PYTHON_LIBRARIES} pybind11::embed )

  target_compile_definitions( test-viame-iqr_session_adaboost PRIVATE
    "VIAME_PYTHON_PACKAGES=\"${VIAME_PYTHON_PACKAGES}\"" )
endif()

target_compile_definitions( test-viame-iqr_session_adaboost PRIVATE
  "VIAME_IQR_ADABOOST_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"" )
