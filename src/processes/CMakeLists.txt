project( kwiver_processes )

SET( sources
  kwiver-config.h
  register_processes.h                      register_processes.cxx
  kw_archive_writer_process.h               kw_archive_writer_process.cxx
)

add_library( processes ${sources} )

# depends on VXL sprokit maptk

target_link_libraries(processes
  LINK_PRIVATE
    sprokit_pipeline
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    vil vul )

