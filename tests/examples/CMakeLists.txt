# VIAME Examples Tests
#
# Integration tests for VIAME example scripts.
# These tests run the actual example scripts and verify they produce output.

if( VIAME_ENABLE_PYTHON )
  # Find pytest
  find_program( PYTEST_EXECUTABLE
    NAMES pytest py.test pytest-3
    DOC "pytest executable"
  )

  if( PYTEST_EXECUTABLE )
    # Object detection example tests
    add_test(
      NAME viame_test_object_detection_examples
      COMMAND ${PYTEST_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/test_object_detection.py"
        -v
        --tb=short
      WORKING_DIRECTORY "${VIAME_SOURCE_DIR}/examples/object_detection"
    )

    set_tests_properties( viame_test_object_detection_examples
      PROPERTIES
        LABELS "VIAME;EXAMPLES;INTEGRATION"
        TIMEOUT 600
    )
  endif()
endif()
