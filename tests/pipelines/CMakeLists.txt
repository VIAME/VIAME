# VIAME Examples Tests
#
# Integration tests for VIAME example scripts.
# These tests run the actual example scripts and verify they produce output.
#
# Test Categories:
#   PIPELINES - Standard pipeline script tests
#   UNIT     - Unit tests (outside pipelines tree)
#
# To run only pipelines tests:
#   ctest -L PIPELINES

if (VIAME_ENABLE_PYTHON)
  # Use Python interpreter with -m pytest for reliable execution
  if (NOT Python3_EXECUTABLE)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
  endif ()

  # Common test properties
  set(PIPELINE_TEST_TIMEOUT 600)

  # Labels for test categories
  set(PIPELINES_TEST_LABELS "PIPELINES")

  # Set up Python path for tests - use the actual install prefix
  set(VIAME_INSTALL_DIR "${VIAME_BUILD_INSTALL_PREFIX}")
  set(VIAME_PYTHON_PATH "${VIAME_INSTALL_DIR}/python")
  set(VIAME_PYTHON_SITE_PACKAGES "${VIAME_INSTALL_DIR}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")

  # PYTHONPATH separator: ":" on Linux, escaped "\;" on Windows (literal ";")
  if (WIN32)
    set(PYPATH_SEP "\;")
  else ()
    set(PYPATH_SEP ":")
  endif ()

  # Helper function to add a standard pipeline script test
  function(add_pipeline_script_test TEST_NAME TEST_FILE TEST_CLASS)
    add_test(
      NAME ${TEST_NAME}
      COMMAND ${Python3_EXECUTABLE} -m pytest
      "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_FILE}"
      -v
      --tb=short
      -k "${TEST_CLASS}"
    )
    set_tests_properties(${TEST_NAME}
      PROPERTIES
      LABELS "${PIPELINES_TEST_LABELS}"
      TIMEOUT ${PIPELINE_TEST_TIMEOUT}
      ENVIRONMENT "PYTHONPATH=${VIAME_PYTHON_PATH}${PYPATH_SEP}${VIAME_PYTHON_SITE_PACKAGES}${PYPATH_SEP}${CMAKE_CURRENT_SOURCE_DIR}${PYPATH_SEP}$ENV{PYTHONPATH};VIAME_INSTALL=${VIAME_INSTALL_DIR}"
    )
  endfunction()

  # ============ Utility Pipelines ============ #

  add_pipeline_script_test(
    viame_pipelines:test_utility_add_head_tail_keypoints_from_dets
    test_utility.py
    TestAddHeadTailKeypointsFromDets
  )

  # ============ Detector Pipelines ============ #

  add_pipeline_script_test(
    viame_pipelines:test_detector_simple_hough
    test_detector.py
    TestSimpleHough
  )

  add_pipeline_script_test(
    viame_pipelines:test_detector_calibration_target
    test_detector.py
    TestCalibrationTarget
  )

  add_pipeline_script_test(
    viame_pipelines:test_default_fish
    test_detector.py
    TestDefaultFish
  )
endif ()