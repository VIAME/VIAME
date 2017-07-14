project(sprokit_processes_examples)

set(examples_srcs
  any_source_process.cxx
  const_process.cxx
  const_number_process.cxx
  data_dependent_process.cxx
  duplicate_process.cxx
  expect_process.cxx
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
  shared_process.cxx
  skip_process.cxx
  tagged_flow_dependent_process.cxx
  take_number_process.cxx
  take_string_process.cxx
  tunable_process.cxx)

set(examples_private_headers
  any_source_process.h
  const_process.h
  const_number_process.h
  data_dependent_process.h
  duplicate_process.h
  examples-config.h
  expect_process.h
  feedback_process.h
  flow_dependent_process.h
  multiplication_process.h
  multiplier_cluster.h
  mutate_process.h
  number_process.h
  orphan_cluster.h
  orphan_process.h
  print_number_process.h
  shared_process.h
  skip_process.h
  tagged_flow_dependent_process.h
  take_number_process.h
  take_string_process.h
  tunable_process.h)

set(no_install FALSE)

sprokit_private_header_group(${examples_private_headers})
sprokit_add_plugin(processes_examples
  MAKE_SPROKIT_PROCESSES_EXAMPLES_LIB
  ${examples_srcs}
  ${examples_private_headers})
target_link_libraries(processes_examples
  PRIVATE        sprokit_pipeline
                 ${Boost_FILESYSTEM_LIBRARY}
                 ${Boost_SYSTEM_LIBRARY}
                 vital_vpm
  )
