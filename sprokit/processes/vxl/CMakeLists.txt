#
# KWIVER processes
#
project( kwiver_processes )

set( sources
  register_processes.cxx

  kw_archive_writer_process.cxx
)

set( private_headers
  kw_archive_writer_process.h
)

kwiver_private_header_group( ${private_headers} )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

kwiver_add_plugin( kwiver_processes_vxl
  SUBDIR          ${kwiver_plugin_process_subdir}
  SOURCES         ${sources}
                  ${private_headers}
  PRIVATE         sprokit_pipeline
                  kwiver_algo_core kwiver_algo_vxl
                  vital vital_vpm vital_logger vital_config
                  vsl vul vnl_algo vnl_io vnl vil_io vil
)
