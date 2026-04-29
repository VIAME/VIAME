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

  set(VIAME_INSTALL_DIR "${VIAME_BUILD_INSTALL_PREFIX}")
  set(VIAME_PY_PATH "${VIAME_INSTALL_DIR}/python")
  set(VIAME_SITE_PACKAGES "${VIAME_INSTALL_DIR}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")

  if (WIN32)
    set(SEP "\;")
  else ()
    set(SEP ":")
  endif ()

  set(INTERNAL_PYTHONPATH "${VIAME_PY_PATH}${SEP}${VIAME_SITE_PACKAGES}${SEP}${CMAKE_CURRENT_SOURCE_DIR}")

  # Common test properties
  set(PIPELINE_TEST_TIMEOUT 600)
  set(PIPELINES_TEST_LABELS "PIPELINES")

  # Helper function to add a standard pipeline script test
  function(add_pipeline_script_test TEST_NAME TEST_FILE TEST_CLASS)
    add_test(
      NAME "viame_pipelines:${TEST_NAME}"
      COMMAND ${Python3_EXECUTABLE} -m pytest "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_FILE}" -v --tb=short -k "${TEST_CLASS}"
    )
    set_tests_properties("viame_pipelines:${TEST_NAME}" PROPERTIES
      LABELS "${PIPELINES_TEST_LABELS}"
      TIMEOUT ${PIPELINE_TEST_TIMEOUT}
      SKIP_RETURN_CODE 5
      ENVIRONMENT "PYTHONPATH=${INTERNAL_PYTHONPATH}${SEP}$ENV{PYTHONPATH};VIAME_INSTALL=${VIAME_INSTALL_DIR}"
    )
    set(options "")
    set(oneValueArgs REQ_ADDON)
    set(multiValueArgs "")
    cmake_parse_arguments(TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(TEST_REQ_ADDON)
      if(NOT ${${TEST_REQ_ADDON}})
        set_tests_properties("viame_pipelines:${TEST_NAME}" PROPERTIES DISABLED TRUE)
      endif()
    endif()
  endfunction()

  # ============ Utility Pipelines ============ #

  add_pipeline_script_test(
    test_utility_add_head_tail_keypoints_from_dets
    test_utility.py
    TestUtilityAddHeadTailKeypointsFromDets
  )

  # ============ Detector Pipelines ============ #

  add_pipeline_script_test(
    test_detector_calibration_target
    test_detector.py
    TestDetectorCalibrationTarget
  )

  add_pipeline_script_test(
    test_detector_generic_proposals
    test_detector.py
    TestDetectorGenericProposals
    REQ_ADDON VIAME_DOWNLOAD_MODELS-GENERIC
  )

  add_pipeline_script_test(
    test_detector_huggingface_zeroshot
    test_detector.py
    TestDetectorHuggingfaceZeroshot
  )

  add_pipeline_script_test(
    test_detector_simple_hough
    test_detector.py
    TestDetectorSimpleHough
  )

  add_pipeline_script_test(
    test_detector_arctic_seal_eo_yolo
    test_detector.py
    TestDetectorArcticSealEOYolo
    REQ_ADDON VIAME_DOWNLOAD_MODELS-ARCTIC-SEAL
  )

  add_pipeline_script_test(
    test_detector_arctic_seal_ir_yolo
    test_detector.py
    TestDetectorArcticSealIRYolo
    REQ_ADDON VIAME_DOWNLOAD_MODELS-ARCTIC-SEAL
  )

  add_pipeline_script_test(
    test_detector_community_fish
    test_detector.py
    TestDetectorCommunityFish
    REQ_ADDON VIAME_DOWNLOAD_MODELS-COMMUNITY-FISH
  )

  add_pipeline_script_test(
    test_detector_default_fish
    test_detector.py
    TestDetectorDefaultFish
    REQ_ADDON VIAME_DOWNLOAD_MODELS-DEFAULT-FISH
  )
endif ()