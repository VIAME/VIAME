include(kwiver-test-setup)

if(KWIVER_ENABLE_FFMPEG)
  add_test(NAME python:core:applets:dump_klv
    COMMAND ${Python_EXECUTABLE}  ${KWIVER_BINARY_DIR}/bin/dump_klv.py
              --config "${kwiver_test_data_directory}/config_files/dump_klv_testing.conf"
              "${kwiver_test_data_directory}/videos/aphill_klv_10s.ts")

  # generate klv_json dump of the video and compare it to reference file
  add_test(NAME python:core:applets:dump_klv_2
    COMMAND ${Python_EXECUTABLE}  ${KWIVER_BINARY_DIR}/bin/dump_klv.py
              --config "${kwiver_test_data_directory}/config_files/dump_klv_testing.conf"
              -l ${KWIVER_BINARY_DIR}/Testing/Temporary/out.json -e klv-json --compress
              "${kwiver_test_data_directory}/videos/aphill_klv_10s.ts")

  add_test(NAME python:core:applets:dump_klv_2:validate
    COMMAND ${CMAKE_COMMAND} -E compare_files
            "${KWIVER_BINARY_DIR}/Testing/Temporary/out.json"
             "${kwiver_test_data_directory}/dump_klv_aphill_output.json")

  # set the dependency between the two test
  set_tests_properties(python:core:applets:dump_klv_2
    PROPERTIES
      FIXTURES_SETUP produce_log)

  set_tests_properties(python:core:applets:dump_klv_2:validate
    PROPERTIES
      FIXTURES_REQUIRED produce_log)

endif()
