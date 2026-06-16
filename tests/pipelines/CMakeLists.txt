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
  set(VIAME_INSTALL_DIR "${VIAME_BUILD_INSTALL_PREFIX}")
  set(VIAME_PY_PATH "${VIAME_INSTALL_DIR}/python")
  set(VIAME_SITE_PACKAGES "${VIAME_INSTALL_DIR}/lib/python3.10/site-packages")

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
      COMMAND ${PYTHON_EXECUTABLE} -m pytest "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_FILE}::${TEST_CLASS}" -v --tb=short
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