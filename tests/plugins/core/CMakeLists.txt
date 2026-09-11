# VIAME Core Plugin Tests

project( test_viame_core )

include( kwiver-test-setup )

# Include directories for plugin headers
include_directories( ${VIAME_SOURCE_DIR}/plugins/core )

set( test_libraries
  viame_core
  kwiver::vital
  kwiver::vital_algo
  GTest::gtest_main )

kwiver_discover_gtests( viame_core measurement_utilities
  LIBRARIES ${test_libraries} )

kwiver_discover_gtests( viame_core resample_object_tracks
  LIBRARIES ${test_libraries} )

kwiver_discover_gtests( viame_core pair_stereo_detections
  LIBRARIES ${test_libraries} )

kwiver_discover_gtests( viame_core windowed_refiner_utils
  LIBRARIES ${test_libraries} )

kwiver_discover_gtests( viame_core evaluate_models
  LIBRARIES ${test_libraries} )

kwiver_discover_gtests( viame_core viame_csv_tot_once
  LIBRARIES ${test_libraries} )

kwiver_discover_gtests( viame_core utilities_training
  LIBRARIES ${test_libraries} )

kwiver_discover_gtests( viame_core manipulate_pipelines
  LIBRARIES ${test_libraries} )

kwiver_discover_gtests( viame_core label_files
  LIBRARIES ${test_libraries} )

# TODO: Enable when tracks_pairing_from_stereo.h is implemented
# kwiver_discover_gtests( viame_core tracks_pairing_from_stereo LIBRARIES ${test_libraries} )
# target_compile_features( test-viame_core-tracks_pairing_from_stereo PRIVATE cxx_std_17 )
# target_compile_definitions( test-viame_core-tracks_pairing_from_stereo PRIVATE TEST_DATA_DIR="${VIAME_SOURCE_DIR}/tests/data" )

if( VIAME_ENABLE_PYTHON )
  set( VIAME_SETUP_SCRIPT "${VIAME_BUILD_INSTALL_PREFIX}/setup_viame.sh" )

  add_test(
    NAME viame_plugins:core:coco_writers
    COMMAND bash -c "source ${VIAME_SETUP_SCRIPT} && \
      python -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_coco_writers.py -v --tb=short"
    WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
  )
  set_tests_properties( viame_plugins:core:coco_writers
    PROPERTIES
      LABELS "UNIT;CORE"
      TIMEOUT 300
  )

  add_test(
    NAME viame_plugins:core:model_wrap
    COMMAND bash -c "source ${VIAME_SETUP_SCRIPT} && \
      python -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_model_wrap.py -v --tb=short"
    WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
  )
  set_tests_properties( viame_plugins:core:model_wrap
    PROPERTIES
      LABELS "UNIT;CORE"
      TIMEOUT 300
  )
endif()

if( VIAME_ENABLE_PYTHON AND VIAME_ENABLE_OPENCV )
  add_test(
    NAME viame_plugins:core:curved_measurement
    COMMAND bash -c "source ${VIAME_SETUP_SCRIPT} && python -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_curved_measurement.py ${CMAKE_CURRENT_SOURCE_DIR}/test_interactive_stereo_points.py -q"
    WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}" )
  set_tests_properties( viame_plugins:core:curved_measurement
    PROPERTIES LABELS "UNIT;CORE" TIMEOUT 300 )
endif()
