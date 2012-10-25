project(vistk_processes_filelist)

set(filelist_srcs
  filelist_reader_process.cxx
  filelist_writer_process.cxx
  registration.cxx)

set(filelist_private_headers
  filelist-config.h
  filelist_reader_process.h
  filelist_writer_process.h
  registration.h)

vistk_private_header_group(${filelist_private_headers})
add_vistk_plugin(processes_filelist
    MAKE_VISTK_PROCESSES_FILELIST_LIB
  ${filelist_srcs}
  ${filelist_private_headers})
target_link_libraries(processes_filelist
  vistk_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
