project(vistk_processes_examples)

set(examples_srcs
  const_process.cxx
  multiplication_process.cxx
  mutate_process.cxx
  number_process.cxx
  orphan_process.cxx
  print_number_process.cxx
  print_string_process.cxx
  registration.cxx)

set(print_number_process_build_options
  ${boost_filesystem_options})

set_source_files_properties(print_number_process.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${print_number_process_build_options}")

set(print_string_process_build_options
  ${boost_filesystem_options})

set_source_files_properties(print_string_process.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${print_string_process_build_options}")

add_vistk_plugin(processes_examples
  MAKE_VISTK_PROCESSES_EXAMPLES_LIB
  ${examples_srcs})
target_link_libraries(processes_examples
  vistk_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
