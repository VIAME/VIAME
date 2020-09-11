#
# Script to set up environment for python nosetests
#


# results of tests being run will be exported to an Xunit xml file
if (NOSE_RUNNER)

  set(no_install TRUE)
  string(TOLOWER "${CMAKE_PROJECT_NAME}" project_name)

  if (WIN32)

    set(kwiver_test_output_path    "${KWIVER_BINARY_DIR}/bin" )

  else ()
# update these to reflect nose -- output location may want to remain the same
# to make dashboard submission easier
    set(kwiver_test_output_path    "${KWIVER_BINARY_DIR}/tests/bin" )

  endif ()
  set(kwiver_test_working_path    "${CTEST_BINARY_DIRECTORY}" )
  set(kwiver_test_runner ${NOSE_RUNNER}
                         ${mod_dst}
                         --with-xunit
                         --xunit-file=nose_results.xml  )
endif()
