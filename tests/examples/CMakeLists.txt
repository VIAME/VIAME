# VIAME Examples Tests
#
# Integration tests for VIAME example scripts and pipelines.
# These tests verify that example pipelines run successfully and produce output.

if( VIAME_ENABLE_PYTHON )
  # Register pytest-based integration tests
  # These require a built VIAME installation to run

  # Find pytest
  find_program( PYTEST_EXECUTABLE
    NAMES pytest py.test pytest-3
    DOC "pytest executable"
  )

  if( PYTEST_EXECUTABLE )
    # Object detection pipeline tests
    add_test(
      NAME viame_test_object_detection_examples
      COMMAND ${PYTEST_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/test_object_detection.py"
        -v
        --tb=short
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    set_tests_properties( viame_test_object_detection_examples
      PROPERTIES
        LABELS "VIAME;EXAMPLES;INTEGRATION"
        ENVIRONMENT "VIAME_INSTALL=${VIAME_INSTALL_PREFIX}"
        TIMEOUT 600
    )
  endif()
endif()
