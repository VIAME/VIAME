# Descriptor formatting, computation and querying
#
# Nothing here may include VXL.
#
# Imported from kwiver in P5-T04: arrows/core's descriptor request handler,
# and the three descriptor and query processes from sprokit/processes/core.
# The imported code is unchanged apart from its include paths and its export
# macro, which is this library's now; its namespace is still kwiver's, and
# phase 8 is where that changes.

set( plugin_headers
  handle_descriptor_request_core.h
  )

set( plugin_sources
  handle_descriptor_request_core.cxx
  )

kwiver_install_headers(
  SUBDIR     viame/descriptors
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_descriptors_export.h
  NOPATH   SUBDIR     viame/descriptors
  )

kwiver_add_library( viame_descriptors
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_descriptors PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

target_link_libraries( viame_descriptors
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
                       kwiver::kwiversys
  )

set_target_properties( viame_descriptors PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

algorithms_create_plugin( viame_descriptors
  register_algorithms.cxx
  )

target_link_libraries( viame_descriptors_plugin
  PUBLIC               kwiver::vital_vpm
  )

###
# Processes
##
set( process_headers
  compute_track_descriptors_process.h
  format_images_srm_process.h
  handle_descriptor_request_process.h
  perform_query_process.h
)

set( process_sources
  register_processes.cxx
  compute_track_descriptors_process.cxx
  format_images_srm_process.cxx
  handle_descriptor_request_process.cxx
  perform_query_process.cxx
)

kwiver_add_plugin( viame_processes_descriptors
  SUBDIR           ${viame_plugin_process_subdir}
  SOURCES          ${process_sources}
                   ${process_headers}
  PRIVATE          kwiver::sprokit_pipeline
                   kwiver::kwiversys
                   kwiver::kwiver_adapter
                   kwiver::vital
                   kwiver::vital_vpm
                   kwiver::vital_logger
                   kwiver::vital_config
                   kwiver::vital_util
                   kwiver::vital_exceptions
                   viame_image_ops
                   viame_descriptors
 )

target_include_directories( viame_processes_descriptors PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
