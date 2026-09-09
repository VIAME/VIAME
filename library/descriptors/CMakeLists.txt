# Descriptor formatting and indexing processes
#
# Nothing here may include VXL.

set( process_headers
  format_images_srm_process.h
)

set( process_sources
  register_processes.cxx
  format_images_srm_process.cxx
)

kwiver_add_plugin( viame_processes_descriptors
  SUBDIR           ${viame_plugin_process_subdir}
  SOURCES          ${process_sources}
                   ${process_headers}
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
                   viame_image_ops
 )

target_include_directories( viame_processes_descriptors PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
