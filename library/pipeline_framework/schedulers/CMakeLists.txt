# The schedulers
#
# `sync` and `thread_per_process` came from kwiver's `sprokit/src/schedulers`
# in P5-T03; `thread_pool`, under `examples/`, in P5-T05. All three are named
# in the registry baseline.

set( source
  registration.cxx
  sync_scheduler.cxx
  thread_per_process_scheduler.cxx
  )

set( private_headers
  sync_scheduler.h
  thread_per_process_scheduler.h
  ${CMAKE_CURRENT_BINARY_DIR}/schedulers_export.h
  )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

kwiver_private_header_group( ${private_headers} )

kwiver_add_plugin( schedulers
  SOURCES         ${source}
                  ${private_headers}
  PRIVATE         sprokit_pipeline
                  vital_config
                  vital_vpm
                  ${CMAKE_THREAD_LIBS_INIT}
  SUBDIR          ${kwiver_plugin_process_subdir}
  )

add_subdirectory( examples )
