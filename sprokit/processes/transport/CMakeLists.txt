project( kwiver_processes_transport )

if ( KWIVER_ENABLE_ZeroMQ )
  set(zmq_sources
    zmq_transport_receive_process.cxx
    zmq_transport_send_process.cxx
    )

  set(zmq_headers
    zmq_transport_receive_process.h
    zmq_transport_send_process.h
    )
  include(kwiver-depends-ZeroMQ)
endif ( KWIVER_ENABLE_ZeroMQ )

set(sources
  register_processes.cxx
  file_transport_send_process.cxx
  ${zmq_sources}
  )

set(private_headers
  file_transport_send_process.h
  ${zmq_headers}
  )

kwiver_private_header_group( ${private_headers} )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

kwiver_add_plugin( kwiver_processes_transport
  SUBDIR      ${kwiver_plugin_process_subdir}
  SOURCES     ${sources}
              ${private_headers}
  PRIVATE     sprokit_pipeline
              vital vital_vpm
              ${ZeroMQ_LIBRARY}
  )

if ( KWIVER_ENABLE_ZeroMQ )
  target_compile_definitions(
    kwiver_processes_transport
    PRIVATE WITH_ZMQ )
endif()
