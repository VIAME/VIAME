# Processes that belong to the pipeline framework itself
#
# Imported from kwiver with sprokit in P5-T03; registered here rather than by
# kwiver's `kwiver_processes` plugin since P5-T04.

set( process_headers
  downsample_process.h
  )

set( process_sources
  register_processes.cxx
  downsample_process.cxx
  )

kwiver_add_plugin( viame_processes_pipeline_framework
  SUBDIR           ${viame_plugin_process_subdir}
  SOURCES          ${process_sources}
                   ${process_headers}
  PRIVATE          kwiver::sprokit_pipeline
                   kwiver::kwiversys
                   kwiver::vital
                   kwiver::vital_vpm
                   kwiver::vital_logger
                   kwiver::vital_config
                   kwiver::vital_util
                   kwiver::vital_exceptions
  )

target_include_directories( viame_processes_pipeline_framework PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
