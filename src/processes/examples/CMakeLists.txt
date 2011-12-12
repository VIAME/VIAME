project(vistk_processes_examples)

set(examples_srcs
  multiplication_process.cxx
  number_process.cxx
  registration.cxx)

add_vistk_plugin(processes_examples
  MAKE_VISTK_PROCESSES_EXAMPLES_LIB
  ${examples_srcs})
target_link_libraries(processes_examples
  vistk_pipeline_types
  vistk_pipeline)
