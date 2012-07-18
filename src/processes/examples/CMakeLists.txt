project(vistk_processes_examples)

set(examples_srcs
  const_process.cxx
  const_number_process.cxx
  data_dependent_process.cxx
  feedback_process.cxx
  flow_dependent_process.cxx
  multiplication_process.cxx
  mutate_process.cxx
  number_process.cxx
  orphan_cluster.cxx
  orphan_process.cxx
  print_number_process.cxx
  registration.cxx
  tagged_flow_dependent_process.cxx
  take_number_process.cxx
  take_string_process.cxx)

set(suppress_install TRUE)

add_vistk_plugin(processes_examples
  MAKE_VISTK_PROCESSES_EXAMPLES_LIB
  ${examples_srcs})
target_link_libraries(processes_examples
  vistk_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
