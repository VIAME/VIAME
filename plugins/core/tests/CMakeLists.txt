project(test_viame_core_plugin)

include(kwiver-test-setup)

set(test_libraries
  viame_core
  GTest::gtest_main)

kwiver_discover_gtests(viame_core_plugin tracks_pairing_from_stereo LIBRARIES ${test_libraries})
target_compile_features(test-viame_core_plugin-tracks_pairing_from_stereo PRIVATE cxx_std_17)
target_compile_definitions(test-viame_core_plugin-tracks_pairing_from_stereo PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
