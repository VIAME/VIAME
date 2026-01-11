# VIAME PyTorch Plugin Tests
#
# Tests for SRNN tracker and pytorch descriptor processes.

if( VIAME_ENABLE_PYTHON )
  # Use Python interpreter with -m pytest for reliable execution
  if( NOT Python3_EXECUTABLE )
    find_package( Python3 COMPONENTS Interpreter REQUIRED )
  endif()

  # Set up Python path for tests - use the actual install prefix
  set( VIAME_INSTALL_DIR "${VIAME_BUILD_INSTALL_PREFIX}" )
  set( VIAME_PYTHON_PATH "${VIAME_INSTALL_DIR}/python" )
  set( VIAME_PYTHON_SITE_PACKAGES "${VIAME_INSTALL_DIR}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages" )

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

endif()
