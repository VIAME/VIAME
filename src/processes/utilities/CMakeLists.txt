project(vistk_processes_utilities)

set(utilities_srcs
  homography_reader_process.cxx
  registration.cxx
  timestamper_process.cxx)

set(homography_reader_process_build_options
  ${boost_filesystem_options})

set_source_files_properties(homography_reader_process.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${homography_reader_process_build_options}")

add_vistk_plugin(processes_utilities
  MAKE_VISTK_PROCESSES_UTILITIES_LIB
  ${utilities_srcs})
target_link_libraries(processes_utilities
  vistk_utilities
  vistk_pipeline
  vgl_algo
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
