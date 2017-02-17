###
# process instrumentation plugins
#
set( sources
  register_plugin.cxx
  logger_process_instrumentation.cxx
  )

set( headers
  logger_process_instrumentation.h
  )

include_directories ( "${CMAKE_CURRENT_BINARY_DIR}" )


kwiver_add_plugin( instrumentation_plugin
  SOURCES          ${sources}
                   ${headers}
                   ${CMAKE_CURRENT_BINARY_DIR}/instrumentation_plugin_export.h
  PRIVATE          vital
                   vital_vpm
                   vital_logger
                   sprokit_pipeline
  SUBDIR           modules
  )
