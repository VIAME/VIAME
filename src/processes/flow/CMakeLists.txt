project(vistk_processes_flow)

set(flow_srcs
  collate_process.cxx
  distribute_process.cxx
  registration.cxx
  sink_process.cxx)

set(flow_private_headers
  collate_process.h
  distribute_process.h
  flow-config.h
  registration.h
  sink_process.h)

add_vistk_plugin(processes_flow
  MAKE_VISTK_PROCESSES_FLOW_LIB
  ${flow_srcs}
  ${flow_private_headers})
target_link_libraries(processes_flow
  vistk_pipeline)
