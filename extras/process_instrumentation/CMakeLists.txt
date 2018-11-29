###
# process instrumentation plugins
#
set( sources
  register_plugin.cxx
  logger_process_instrumentation.cxx
  timing_process_instrumentation.cxx
  )

set( headers
  logger_process_instrumentation.h
  timing_process_instrumentation.h
  )

include_directories ( "${CMAKE_CURRENT_BINARY_DIR}" )


kwiver_add_plugin( instrumentation_plugin
  SOURCES          ${sources}
                   ${headers}
                   ${CMAKE_CURRENT_BINARY_DIR}/instrumentation_plugin_export.h
  PRIVATE          vital
                   vital_vpm
                   vital_util
                   vital_logger
                   sprokit_pipeline
  SUBDIR           ${kwiver_plugin_process_instrumentation_subdir}
  )


if (KWIVER_ENABLE_TOOLS)
###
#     plugins for plugin explorer
kwiver_add_plugin( process_instrumentation_plugin
  SUBDIR   ${kwiver_plugin_plugin_explorer_subdir}
  SOURCES  process_instrumentation_plugin.cxx
  PRIVATE  vital
           vital_vpm
           vital_config
           vital_algo
           explorer_plugin
           sprokit_pipeline)
endif()
