project(sprokit_test_processes)

include(kwiver-test-setup)

set( test_libraries       vital sprokit_pipeline sprokit_pipeline_util kwiver_adapter )

#############################
# adapter process tests
#############################

#kwiver_discover_tests(adapter_basic     LIBRARIES ${test_libraries} )
#kwiver_discover_tests(non_blocking      LIBRARIES ${test_libraries} )

set( private_headers
  test_proc_seq.h
  )

set( sources
  register_processes.cxx

  test_proc_seq.cxx
  )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

kwiver_add_plugin( kwiver_processes_test
  SUBDIR           ${kwiver_plugin_process_subdir}
  SOURCES          ${sources}
                   ${private_headers}
  PRIVATE          sprokit_pipeline
                   vital vital_vpm vital_logger vital_config
                   kwiversys
 )
