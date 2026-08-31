# VIAME Pipeline Tests
#
# Integration tests that run every DIVE-runnable pipeline the way DIVE runs it.
# One ctest per pipeline category; the pipelines within a category are
# discovered from the install tree at test time (see cases.py), so add-on
# pipelines are covered without being named here.
#
# To run only pipelines tests:
#   ctest -L PIPELINES
# To run a single pipeline:
#   pytest tests/pipelines/test_detector.py -k detector_default_fish

if (VIAME_ENABLE_PYTHON)
  # Generous: a category can hold dozens of model pipelines on a full install.
  set(PIPELINE_TEST_TIMEOUT 7200)

  DownloadAndExtract(
    https://data.kitware.com/api/v1/file/6a311e9ba7bcc12a3cf54ac1/download
    4f89aea0e8fc4323bc48557d3efe5e98
    ${VIAME_DOWNLOAD_DIR}/pipelines_test_data.zip
    ${CMAKE_CURRENT_SOURCE_DIR} )

  foreach (category detector tracker filter transcode utility measurement)
    viame_add_pytest_test(
      NAME "viame_pipelines:test_${category}"
      TARGET "${CMAKE_CURRENT_SOURCE_DIR}/test_${category}.py"
      LABELS "PIPELINES"
      TIMEOUT "${PIPELINE_TEST_TIMEOUT}"
      SKIP_RETURN_CODE 5
      PYTHONPATH_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
      SOURCE_SETUP ON
    )
  endforeach ()

endif ()
