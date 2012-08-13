project(vistk_processes_utilities)

set(utilities_srcs
  homography_reader_process.cxx
  registration.cxx
  timestamp_reader_process.cxx
  timestamper_process.cxx)

set(utilities_private_headers
  homography_reader_process.h
  registration.h
  timestamp_reader_process.h
  timestamper_process.h
  utilities-config.h)

vistk_private_header_group(${utilities_private_headers})
add_vistk_plugin(processes_utilities
  MAKE_VISTK_PROCESSES_UTILITIES_LIB
  ${utilities_srcs}
  ${utilities_private_headers})
target_link_libraries(processes_utilities
  vistk_utilities
  vistk_pipeline
  vgl_algo
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
