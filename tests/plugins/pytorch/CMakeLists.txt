# VIAME PyTorch Plugin Tests
#
# Tests for SRNN tracker, pytorch descriptor processes, and SAM3.
# These source setup_viame.sh (SOURCE_SETUP) so native torch/kwiver modules
# resolve at import time.

if( VIAME_ENABLE_PYTHON )
  set( VIAME_INSTALL_DIR "${VIAME_BUILD_INSTALL_PREFIX}" )

  # SRNN Tracker and Descriptor Tests
  viame_add_pytest_test(
    NAME viame_plugins:pytorch:srnn_tracker
    TARGET "${CMAKE_CURRENT_SOURCE_DIR}/test_processes.py" -k test_srnn_tracker
    LABELS UNIT PYTORCH
    TIMEOUT 600
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
  )

  if( VIAME_ENABLE_PYTORCH-VISION )
    viame_add_pytest_test(
      NAME viame_plugins:pytorch:alexnet_descriptors
      TARGET "${CMAKE_CURRENT_SOURCE_DIR}/test_processes.py" -k test_alexnet_descriptors
      LABELS UNIT PYTORCH
      TIMEOUT 600
      SOURCE_SETUP
      WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
    )
    viame_add_pytest_test(
      NAME viame_plugins:pytorch:resnet_descriptors
      TARGET "${CMAKE_CURRENT_SOURCE_DIR}/test_processes.py" -k test_resnet_descriptors
      LABELS UNIT PYTORCH
      TIMEOUT 600
      SOURCE_SETUP
      WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
    )
    viame_add_pytest_test(
      NAME viame_plugins:pytorch:desc_augmentation
      TARGET "${CMAKE_CURRENT_SOURCE_DIR}/test_processes.py" -k test_desc_augmentation
      LABELS UNIT PYTORCH
      TIMEOUT 600
      SOURCE_SETUP
      WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
    )
  endif()

  # ============================================================================
  # SAM3 Tests
  # ============================================================================
  viame_add_pytest_test(
    NAME viame_plugins:pytorch:sam3_utilities
    TARGET "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py" -k TestSAM3Utilities
    LABELS UNIT PYTORCH SAM3
    TIMEOUT 300
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
  )
  viame_add_pytest_test(
    NAME viame_plugins:pytorch:sam3_model_manager
    TARGET "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py" -k TestSAM3ModelManager
    LABELS UNIT PYTORCH SAM3
    TIMEOUT 300
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
  )
  viame_add_pytest_test(
    NAME viame_plugins:pytorch:sam3_refiner
    TARGET "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py" -k TestSAM3Refiner
    LABELS UNIT PYTORCH SAM3
    TIMEOUT 300
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
  )
  viame_add_pytest_test(
    NAME viame_plugins:pytorch:sam3_pipelines
    TARGET "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py" -k TestSAM3Pipelines
    LABELS INTEGRATION PYTORCH SAM3
    TIMEOUT 900
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
  )
  viame_add_pytest_test(
    NAME viame_plugins:pytorch:sam3_text_queries
    TARGET "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py" -k TestSAM3TextQueries
    LABELS INTEGRATION PYTORCH SAM3
    TIMEOUT 600
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
  )
  viame_add_pytest_test(
    NAME viame_plugins:pytorch:sam3_point_click
    TARGET "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py" -k TestSAM3PointClickSegmentation
    LABELS INTEGRATION PYTORCH SAM3
    TIMEOUT 600
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
  )
  viame_add_pytest_test(
    NAME viame_plugins:pytorch:sam3_registration
    TARGET "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py" -k TestSAM3AlgorithmRegistration
    LABELS UNIT PYTORCH SAM3
    TIMEOUT 300
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
  )
  viame_add_pytest_test(
    NAME viame_plugins:pytorch:sam3_integration
    TARGET "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py" -k TestSAM3Integration
    LABELS INTEGRATION PYTORCH SAM3
    TIMEOUT 900
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
  )

  # Run all SAM3 tests together (convenience target)
  viame_add_pytest_test(
    NAME viame_plugins:pytorch:sam3_all
    TARGET "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py"
    LABELS INTEGRATION PYTORCH SAM3
    TIMEOUT 1800
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
  )

endif()
