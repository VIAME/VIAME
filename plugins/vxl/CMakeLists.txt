# Build / Install Plugin containing VXL Algorithm Implementations

# Add process library
set( process_headers
  vxl_srm_image_formatter_process.h
)

set( process_sources
  register_processes.cxx
  vxl_srm_image_formatter_process.cxx
)

kwiver_add_plugin( viame_processes_core
  SUBDIR           ${viame_plugin_process_subdir}
  SOURCES          ${process_sources}
                   ${private_headers}
  PRIVATE          kwiver::sprokit_pipeline
                   kwiver::kwiver_algo_core
                   kwiver::kwiversys
                   kwiver::kwiver_adapter
                   kwiver::vital
                   kwiver::vital_vpm
                   kwiver::vital_logger
                   kwiver::vital_config
                   kwiver::vital_util
                   kwiver::vital_exceptions
                   ${Boost_SYSTEM_LIBRARY}
                   ${Boost_FILESYSTEM_LIBRARY}
 )
