project(kwiver_processes_transport)

set(sources
  register_processes.cxx

  file_transport_send_process.cxx
  )

set(private_headers

  file_transport_send_process.h
  )

kwiver_private_header_group( ${private_headers} )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

kwiver_add_plugin(processes_transport
  SUBDIR      sprokit
  SOURCES     ${sources}
              ${private_headers}
  PRIVATE     sprokit_pipeline
  )
