# VIAME Core Plugin Tests

project( test_viame_core )

include( kwiver-test-setup )

# Include directories for plugin headers
include_directories( ${VIAME_SOURCE_DIR}/plugins/core )

set( test_libraries
  viame_core
  GTest::gtest_main )

kwiver_discover_gtests( viame_core measurement_utilities
  LIBRARIES ${test_libraries}
  PROPERTIES LABELS "VIAME;UNIT" )

# TODO: Enable when tracks_pairing_from_stereo.h is implemented
# kwiver_discover_gtests( viame_core tracks_pairing_from_stereo LIBRARIES ${test_libraries} )
# target_compile_features( test-viame_core-tracks_pairing_from_stereo PRIVATE cxx_std_17 )
# target_compile_definitions( test-viame_core-tracks_pairing_from_stereo PRIVATE TEST_DATA_DIR="${VIAME_SOURCE_DIR}/tests/data" )
