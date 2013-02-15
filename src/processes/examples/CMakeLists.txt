project(vistk_processes_examples)

set(examples_srcs
  const_process.cxx
  const_number_process.cxx
  data_dependent_process.cxx
  feedback_process.cxx
  flow_dependent_process.cxx
  multiplication_process.cxx
  multiplier_cluster.cxx
  mutate_process.cxx
  number_process.cxx
  orphan_cluster.cxx
  orphan_process.cxx
  print_number_process.cxx
  registration.cxx
  tagged_flow_dependent_process.cxx
  take_number_process.cxx
  take_string_process.cxx)

set(examples_private_headers
  const_process.h
  const_number_process.h
  data_dependent_process.h
  examples-config.h
  feedback_process.h
  flow_dependent_process.h
  multiplication_process.h
  multiplier_cluster.h
  mutate_process.h
  number_process.h
  orphan_cluster.h
  orphan_process.h
  print_number_process.h
  registration.h
  tagged_flow_dependent_process.h
  take_number_process.h
  take_string_process.h)

set(suppress_install TRUE)

vistk_private_header_group(${examples_private_headers})
add_vistk_plugin(processes_examples
  MAKE_VISTK_PROCESSES_EXAMPLES_LIB
  ${examples_srcs}
  ${examples_private_headers})
target_link_libraries(processes_examples
  LINK_PRIVATE
    vistk_pipeline
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY})
