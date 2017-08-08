project(sprokit_processes_flow)

set(flow_srcs
  collate_process.cxx
  distribute_process.cxx
  pass_process.cxx
  registration.cxx
  sink_process.cxx)

set(flow_private_headers
  collate_process.h
  distribute_process.h
  pass_process.h
  sink_process.h)

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

sprokit_private_header_group(${flow_private_headers})
kwiver_add_plugin(processes_flow
  SUBDIR      sprokit
  SOURCES     ${flow_srcs}
              ${flow_private_headers}
  PRIVATE     sprokit_pipeline
              vital_vpm
  )
