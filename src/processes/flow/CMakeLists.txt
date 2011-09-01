project(vistk_processes_flow)

set(flow_srcs
  collate_process.cxx
  distribute_process.cxx
  registration.cxx
  sink_process.cxx
  source_process.cxx)

add_vistk_plugin(processes_flow
  MAKE_VISTK_PROCESSES_FLOW_LIB
  ${flow_srcs})
target_link_libraries(processes_flow
  vistk_pipeline)
