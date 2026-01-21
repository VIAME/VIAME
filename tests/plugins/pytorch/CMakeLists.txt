# VIAME PyTorch Plugin Tests
#
# Tests for SRNN tracker, pytorch descriptor processes, and SAM3.

if( VIAME_ENABLE_PYTHON )
  # Set up paths for tests
  set( VIAME_INSTALL_DIR "${VIAME_BUILD_INSTALL_PREFIX}" )
  set( VIAME_SETUP_SCRIPT "${VIAME_INSTALL_DIR}/setup_viame.sh" )

  # Helper macro to add a pytest test that sources setup_viame.sh
  macro( viame_add_pytest_test NAME TEST_FILE TEST_FILTER )
    add_test(
      NAME ${NAME}
      COMMAND bash -c "source ${VIAME_SETUP_SCRIPT} && python -m pytest ${TEST_FILE} -v --tb=short -k ${TEST_FILTER}"
      WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
    )
  endmacro()

  # SRNN Tracker and Descriptor Tests
  viame_add_pytest_test(
    viame_plugins:pytorch:srnn_tracker
    "${CMAKE_CURRENT_SOURCE_DIR}/test_processes.py"
    "test_srnn_tracker"
  )
  set_tests_properties( viame_plugins:pytorch:srnn_tracker
    PROPERTIES
      LABELS "UNIT;PYTORCH"
      TIMEOUT 600
  )

  if( VIAME_ENABLE_PYTORCH-VISION )
    viame_add_pytest_test(
      viame_plugins:pytorch:alexnet_descriptors
      "${CMAKE_CURRENT_SOURCE_DIR}/test_processes.py"
      "test_alexnet_descriptors"
    )
    set_tests_properties( viame_plugins:pytorch:alexnet_descriptors
      PROPERTIES
        LABELS "UNIT;PYTORCH"
        TIMEOUT 600
    )

    viame_add_pytest_test(
      viame_plugins:pytorch:resnet_descriptors
      "${CMAKE_CURRENT_SOURCE_DIR}/test_processes.py"
      "test_resnet_descriptors"
    )
    set_tests_properties( viame_plugins:pytorch:resnet_descriptors
      PROPERTIES
        LABELS "UNIT;PYTORCH"
        TIMEOUT 600
    )

    viame_add_pytest_test(
      viame_plugins:pytorch:desc_augmentation
      "${CMAKE_CURRENT_SOURCE_DIR}/test_processes.py"
      "test_desc_augmentation"
    )
    set_tests_properties( viame_plugins:pytorch:desc_augmentation
      PROPERTIES
        LABELS "UNIT;PYTORCH"
        TIMEOUT 600
    )
  endif()

  # ============================================================================
  # SAM3 Tests
  # ============================================================================

  # SAM3 Utilities Unit Tests
  viame_add_pytest_test(
    viame_plugins:pytorch:sam3_utilities
    "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py"
    "TestSAM3Utilities"
  )
  set_tests_properties( viame_plugins:pytorch:sam3_utilities
    PROPERTIES
      LABELS "UNIT;PYTORCH;SAM3"
      TIMEOUT 300
  )

  # SAM3 Model Manager Tests
  viame_add_pytest_test(
    viame_plugins:pytorch:sam3_model_manager
    "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py"
    "TestSAM3ModelManager"
  )
  set_tests_properties( viame_plugins:pytorch:sam3_model_manager
    PROPERTIES
      LABELS "UNIT;PYTORCH;SAM3"
      TIMEOUT 300
  )

  # SAM3 Refiner Tests
  viame_add_pytest_test(
    viame_plugins:pytorch:sam3_refiner
    "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py"
    "TestSAM3Refiner"
  )
  set_tests_properties( viame_plugins:pytorch:sam3_refiner
    PROPERTIES
      LABELS "UNIT;PYTORCH;SAM3"
      TIMEOUT 300
  )

  # SAM3 Pipeline Tests (requires SAM3 models)
  viame_add_pytest_test(
    viame_plugins:pytorch:sam3_pipelines
    "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py"
    "TestSAM3Pipelines"
  )
  set_tests_properties( viame_plugins:pytorch:sam3_pipelines
    PROPERTIES
      LABELS "INTEGRATION;PYTORCH;SAM3"
      TIMEOUT 900
  )

  # SAM3 Text Query Tests
  viame_add_pytest_test(
    viame_plugins:pytorch:sam3_text_queries
    "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py"
    "TestSAM3TextQueries"
  )
  set_tests_properties( viame_plugins:pytorch:sam3_text_queries
    PROPERTIES
      LABELS "INTEGRATION;PYTORCH;SAM3"
      TIMEOUT 600
  )

  # SAM3 Point Click Segmentation Tests
  viame_add_pytest_test(
    viame_plugins:pytorch:sam3_point_click
    "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py"
    "TestSAM3PointClickSegmentation"
  )
  set_tests_properties( viame_plugins:pytorch:sam3_point_click
    PROPERTIES
      LABELS "INTEGRATION;PYTORCH;SAM3"
      TIMEOUT 600
  )

  # SAM3 Algorithm Registration Tests
  viame_add_pytest_test(
    viame_plugins:pytorch:sam3_registration
    "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py"
    "TestSAM3AlgorithmRegistration"
  )
  set_tests_properties( viame_plugins:pytorch:sam3_registration
    PROPERTIES
      LABELS "UNIT;PYTORCH;SAM3"
      TIMEOUT 300
  )

  # SAM3 Integration Tests (full workflow)
  viame_add_pytest_test(
    viame_plugins:pytorch:sam3_integration
    "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py"
    "TestSAM3Integration"
  )
  set_tests_properties( viame_plugins:pytorch:sam3_integration
    PROPERTIES
      LABELS "INTEGRATION;PYTORCH;SAM3"
      TIMEOUT 900
  )

  # Run all SAM3 tests together (convenience target)
  add_test(
    NAME viame_plugins:pytorch:sam3_all
    COMMAND bash -c "source ${VIAME_SETUP_SCRIPT} && python -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py -v --tb=short"
    WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
  )
  set_tests_properties( viame_plugins:pytorch:sam3_all
    PROPERTIES
      LABELS "INTEGRATION;PYTORCH;SAM3"
      TIMEOUT 1800
  )

endif()
