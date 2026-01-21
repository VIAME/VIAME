# VIAME PyTorch Plugin Tests
#
# Tests for SRNN tracker, pytorch descriptor processes, and SAM3.

if( VIAME_ENABLE_PYTHON )
  # Use Python interpreter with -m pytest for reliable execution
  if( NOT Python3_EXECUTABLE )
    find_package( Python3 COMPONENTS Interpreter REQUIRED )
  endif()

  # Set up Python path for tests - use the actual install prefix
  set( VIAME_INSTALL_DIR "${VIAME_BUILD_INSTALL_PREFIX}" )
  set( VIAME_PYTHON_PATH "${VIAME_INSTALL_DIR}/python" )
  set( VIAME_PYTHON_SITE_PACKAGES "${VIAME_INSTALL_DIR}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages" )

  # Common test environment
  set( PYTORCH_TEST_ENV
    "PYTHONPATH=${VIAME_PYTHON_PATH}:${VIAME_PYTHON_SITE_PACKAGES}:$ENV{PYTHONPATH}"
    "PATH=${VIAME_INSTALL_DIR}/bin:$ENV{PATH}"
    "VIAME_INSTALL=${VIAME_INSTALL_DIR}"
  )

  # SRNN Tracker and Descriptor Tests
  add_test(
    NAME viame_plugins:pytorch:srnn_tracker
    COMMAND ${Python3_EXECUTABLE} -m pytest
      "${CMAKE_CURRENT_SOURCE_DIR}/test_processes.py"
      -v
      --tb=short
      -k "test_srnn_tracker"
    WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
  )
  set_tests_properties( viame_plugins:pytorch:srnn_tracker
    PROPERTIES
      LABELS "UNIT;PYTORCH"
      TIMEOUT 600
      ENVIRONMENT "PYTHONPATH=${VIAME_PYTHON_PATH}:${VIAME_PYTHON_SITE_PACKAGES}:$ENV{PYTHONPATH};PATH=${VIAME_INSTALL_DIR}/bin:$ENV{PATH}"
  )

  if( VIAME_ENABLE_PYTORCH-VISION )
    add_test(
      NAME viame_plugins:pytorch:alexnet_descriptors
      COMMAND ${Python3_EXECUTABLE} -m pytest
        "${CMAKE_CURRENT_SOURCE_DIR}/test_processes.py"
        -v
        --tb=short
        -k "test_alexnet_descriptors"
      WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
    )
    set_tests_properties( viame_plugins:pytorch:alexnet_descriptors
      PROPERTIES
        LABELS "UNIT;PYTORCH"
        TIMEOUT 600
        ENVIRONMENT "PYTHONPATH=${VIAME_PYTHON_PATH}:${VIAME_PYTHON_SITE_PACKAGES}:$ENV{PYTHONPATH};PATH=${VIAME_INSTALL_DIR}/bin:$ENV{PATH}"
    )

    add_test(
      NAME viame_plugins:pytorch:resnet_descriptors
      COMMAND ${Python3_EXECUTABLE} -m pytest
        "${CMAKE_CURRENT_SOURCE_DIR}/test_processes.py"
        -v
        --tb=short
        -k "test_resnet_descriptors"
      WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
    )
    set_tests_properties( viame_plugins:pytorch:resnet_descriptors
      PROPERTIES
        LABELS "UNIT;PYTORCH"
        TIMEOUT 600
        ENVIRONMENT "PYTHONPATH=${VIAME_PYTHON_PATH}:${VIAME_PYTHON_SITE_PACKAGES}:$ENV{PYTHONPATH};PATH=${VIAME_INSTALL_DIR}/bin:$ENV{PATH}"
    )

    add_test(
      NAME viame_plugins:pytorch:desc_augmentation
      COMMAND ${Python3_EXECUTABLE} -m pytest
        "${CMAKE_CURRENT_SOURCE_DIR}/test_processes.py"
        -v
        --tb=short
        -k "test_desc_augmentation"
      WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
    )
    set_tests_properties( viame_plugins:pytorch:desc_augmentation
      PROPERTIES
        LABELS "UNIT;PYTORCH"
        TIMEOUT 600
        ENVIRONMENT "PYTHONPATH=${VIAME_PYTHON_PATH}:${VIAME_PYTHON_SITE_PACKAGES}:$ENV{PYTHONPATH};PATH=${VIAME_INSTALL_DIR}/bin:$ENV{PATH}"
    )
  endif()

  # ============================================================================
  # SAM3 Tests
  # ============================================================================

  # SAM3 Utilities Unit Tests
  add_test(
    NAME viame_plugins:pytorch:sam3_utilities
    COMMAND ${Python3_EXECUTABLE} -m pytest
      "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py"
      -v
      --tb=short
      -k "TestSAM3Utilities"
    WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
  )
  set_tests_properties( viame_plugins:pytorch:sam3_utilities
    PROPERTIES
      LABELS "UNIT;PYTORCH;SAM3"
      TIMEOUT 300
      ENVIRONMENT "${PYTORCH_TEST_ENV}"
  )

  # SAM3 Model Manager Tests
  add_test(
    NAME viame_plugins:pytorch:sam3_model_manager
    COMMAND ${Python3_EXECUTABLE} -m pytest
      "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py"
      -v
      --tb=short
      -k "TestSAM3ModelManager"
    WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
  )
  set_tests_properties( viame_plugins:pytorch:sam3_model_manager
    PROPERTIES
      LABELS "UNIT;PYTORCH;SAM3"
      TIMEOUT 300
      ENVIRONMENT "${PYTORCH_TEST_ENV}"
  )

  # SAM3 Refiner Tests
  add_test(
    NAME viame_plugins:pytorch:sam3_refiner
    COMMAND ${Python3_EXECUTABLE} -m pytest
      "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py"
      -v
      --tb=short
      -k "TestSAM3Refiner"
    WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
  )
  set_tests_properties( viame_plugins:pytorch:sam3_refiner
    PROPERTIES
      LABELS "UNIT;PYTORCH;SAM3"
      TIMEOUT 300
      ENVIRONMENT "${PYTORCH_TEST_ENV}"
  )

  # SAM3 Pipeline Tests (requires SAM3 models)
  add_test(
    NAME viame_plugins:pytorch:sam3_pipelines
    COMMAND ${Python3_EXECUTABLE} -m pytest
      "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py"
      -v
      --tb=short
      -k "TestSAM3Pipelines"
    WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
  )
  set_tests_properties( viame_plugins:pytorch:sam3_pipelines
    PROPERTIES
      LABELS "INTEGRATION;PYTORCH;SAM3"
      TIMEOUT 900
      ENVIRONMENT "${PYTORCH_TEST_ENV}"
  )

  # SAM3 Text Query Tests
  add_test(
    NAME viame_plugins:pytorch:sam3_text_queries
    COMMAND ${Python3_EXECUTABLE} -m pytest
      "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py"
      -v
      --tb=short
      -k "TestSAM3TextQueries"
    WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
  )
  set_tests_properties( viame_plugins:pytorch:sam3_text_queries
    PROPERTIES
      LABELS "INTEGRATION;PYTORCH;SAM3"
      TIMEOUT 600
      ENVIRONMENT "${PYTORCH_TEST_ENV}"
  )

  # SAM3 Point Click Segmentation Tests
  add_test(
    NAME viame_plugins:pytorch:sam3_point_click
    COMMAND ${Python3_EXECUTABLE} -m pytest
      "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py"
      -v
      --tb=short
      -k "TestSAM3PointClickSegmentation"
    WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
  )
  set_tests_properties( viame_plugins:pytorch:sam3_point_click
    PROPERTIES
      LABELS "INTEGRATION;PYTORCH;SAM3"
      TIMEOUT 600
      ENVIRONMENT "${PYTORCH_TEST_ENV}"
  )

  # SAM3 Algorithm Registration Tests
  add_test(
    NAME viame_plugins:pytorch:sam3_registration
    COMMAND ${Python3_EXECUTABLE} -m pytest
      "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py"
      -v
      --tb=short
      -k "TestSAM3AlgorithmRegistration"
    WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
  )
  set_tests_properties( viame_plugins:pytorch:sam3_registration
    PROPERTIES
      LABELS "UNIT;PYTORCH;SAM3"
      TIMEOUT 300
      ENVIRONMENT "${PYTORCH_TEST_ENV}"
  )

  # SAM3 Integration Tests (full workflow)
  add_test(
    NAME viame_plugins:pytorch:sam3_integration
    COMMAND ${Python3_EXECUTABLE} -m pytest
      "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py"
      -v
      --tb=short
      -k "TestSAM3Integration"
    WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
  )
  set_tests_properties( viame_plugins:pytorch:sam3_integration
    PROPERTIES
      LABELS "INTEGRATION;PYTORCH;SAM3"
      TIMEOUT 900
      ENVIRONMENT "${PYTORCH_TEST_ENV}"
  )

  # Run all SAM3 tests together (convenience target)
  add_test(
    NAME viame_plugins:pytorch:sam3_all
    COMMAND ${Python3_EXECUTABLE} -m pytest
      "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py"
      -v
      --tb=short
    WORKING_DIRECTORY "${VIAME_INSTALL_DIR}"
  )
  set_tests_properties( viame_plugins:pytorch:sam3_all
    PROPERTIES
      LABELS "INTEGRATION;PYTORCH;SAM3"
      TIMEOUT 1800
      ENVIRONMENT "${PYTORCH_TEST_ENV}"
  )

endif()
