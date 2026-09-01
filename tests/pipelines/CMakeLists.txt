# VIAME Pipeline Tests
#
# Integration tests that run every DIVE-runnable pipeline the way DIVE runs it.
# One ctest per pipeline, discovered from the install tree when ctest runs
# rather than when CMake configures, so add-on pipelines that arrive later as
# downloaded model packs are covered without being named here.
#
# To run only pipelines tests:
#   ctest -L PIPELINES
# To list every pipeline as its own test:
#   ctest -N -L PIPELINES
# To run a single pipeline:
#   ctest -R viame_pipelines:detector:detector_default_fish

if (VIAME_ENABLE_PYTHON)
  DownloadAndExtract(
    https://data.kitware.com/api/v1/file/6a311e9ba7bcc12a3cf54ac1/download
    4f89aea0e8fc4323bc48557d3efe5e98
    ${VIAME_DOWNLOAD_DIR}/pipelines_test_data.zip
    ${CMAKE_CURRENT_SOURCE_DIR} )

  set( PIPELINE_CATEGORIES detector tracker filter transcode utility measurement )

  # Each generated test runs exactly one pipeline, so the per-category budget
  # that had to cover dozens of them no longer applies.
  set( PIPELINE_CASE_TIMEOUT 1800 )

  # The generated tests source the install setup script exactly as the bundled
  # per-category tests did, so their execution model is unchanged.
  if( WIN32 )
    set( PIPELINE_SETUP_COMMAND "call \"${VIAME_BUILD_INSTALL_PREFIX}/setup_viame.bat\"" )
  else()
    set( PIPELINE_SETUP_COMMAND "source \"${VIAME_BUILD_INSTALL_PREFIX}/setup_viame.sh\"" )
  endif()

  # Discovery alone needs a PYTHONPATH: it imports cases.py outside pytest and
  # so misses the tests/common insertion tests/conftest.py performs.
  set( _pipeline_site_packages
    "${VIAME_BUILD_INSTALL_PREFIX}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages" )
  string( JOIN "${VIAME_TEST_PYPATH_SEP}" PIPELINE_DISCOVERY_PYTHONPATH
    "${VIAME_BUILD_INSTALL_PREFIX}/python"
    "${_pipeline_site_packages}"
    "${VIAME_TESTS_COMMON_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}" )

  set( _pipeline_discovery
    "${CMAKE_CURRENT_BINARY_DIR}/discover_pipeline_tests.cmake" )
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/discover_pipeline_tests.cmake.in"
    "${_pipeline_discovery}"
    @ONLY )

  set_property( DIRECTORY APPEND
    PROPERTY TEST_INCLUDE_FILES "${_pipeline_discovery}" )

endif ()
