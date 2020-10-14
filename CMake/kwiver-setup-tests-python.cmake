#
# Script to set up environment for python nosetests
#


# results of tests being run will be exported to an Xunit xml file
if (NOSE_RUNNER)

  set(no_install TRUE)
  string(TOLOWER "${CMAKE_PROJECT_NAME}" project_name)

  if (WIN32)
    if(VENV_CREATED)
      if(Python3_INTERPRETER_ID STREQUAL "Anaconda")
        set(NOSE_COM "conda' 'activate' 'testing_venv' && ")
      else()
        set(NOSE_COM "'source' '${KWIVER_BINARY_DIR}/testing_venv/bin/activate' && ")
      endif()
    else()
      set(NOSE_COM)
    endif()
    set(kwiver_test_output_path    "${KWIVER_BINARY_DIR}/$<CONFIG>/bin" )
  else ()
    if(VENV_CREATED)
      if(Python3_INTERPRETER_ID STREQUAL "Anaconda")
        set(NOSE_COM "conda' 'activate' 'testing_venv' && ")
      else()
        set(NOSE_COM "'source' '${KWIVER_BINARY_DIR}/testing_venv/bin/activate' && ")
      endif()
    else()
      set(NOSE_COM)
    endif()
    set(kwiver_test_output_path    "${KWIVER_BINARY_DIR}/tests/bin" )
  endif ()
  set(kwiver_test_working_path    "${CTEST_BINARY_DIRECTORY}" )

  set(kwiver_test_runner "${NOSE_RUNNER} '${mod_dst}")
endif()
