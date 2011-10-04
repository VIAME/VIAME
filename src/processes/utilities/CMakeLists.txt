set(utilities_srcs
  registration.cxx)

add_vistk_plugin(processes_utilities
  MAKE_VISTK_PROCESSES_UTILITIES_LIB
  ${utilities_srcs})
target_link_libraries(processes_utilities
  vistk_pipeline)
