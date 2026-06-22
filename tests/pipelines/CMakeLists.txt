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
  # Common test properties
  set(PIPELINE_TEST_TIMEOUT 600)
  set(PIPELINES_TEST_LABELS "PIPELINES")

  # Thin wrapper over the shared viame_add_pytest_test() helper
  # (tests/common/add_pytest_test.cmake). REQ_ADDON gates the test on a
  # model add-on being enabled.
  function(add_pipeline_script_test TEST_NAME TEST_FILE TEST_CLASS)
    set(options "")
    set(oneValueArgs REQ_ADDON)
    set(multiValueArgs "")
    cmake_parse_arguments(TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(disabled_arg "")
    if(TEST_REQ_ADDON AND NOT ${${TEST_REQ_ADDON}})
      set(disabled_arg DISABLED)
    endif()

    viame_add_pytest_test(
      NAME "viame_pipelines:${TEST_NAME}"
      TARGET "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_FILE}::${TEST_CLASS}"
      LABELS ${PIPELINES_TEST_LABELS}
      TIMEOUT ${PIPELINE_TEST_TIMEOUT}
      SKIP_RETURN_CODE 5
      PYTHONPATH_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
      ${disabled_arg}
    )
  endfunction()

  DownloadAndExtract(
    https://data.kitware.com/api/v1/file/6a311e9ba7bcc12a3cf54ac1/download
    4f89aea0e8fc4323bc48557d3efe5e98
    ${VIAME_DOWNLOAD_DIR}/pipelines_test_data.zip
    ${CMAKE_CURRENT_SOURCE_DIR} )

  include(cmake/utility_tests.cmake)
  include(cmake/detector_tests.cmake)
  include(cmake/filter_tests.cmake)
  include(cmake/transcode_tests.cmake)
  include(cmake/measurement_tests.cmake)
  include(cmake/tracker_tests.cmake)

endif ()