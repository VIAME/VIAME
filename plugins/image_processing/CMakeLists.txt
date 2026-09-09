# In-house image processing algorithms
#
# These replace the third-party implementations one at a time, keeping the
# registered names and config keys. Nothing here may include OpenCV or VXL.

set( plugin_headers
  average_frames.h
  color_commonality.h
  core_image_io.h
  convert_image.h
  morphology.h
  threshold.h
  )

set( plugin_sources
  average_frames.cxx
  color_commonality.cxx
  core_image_io.cxx
  convert_image.cxx
  morphology.cxx
  threshold.cxx
  )

kwiver_install_headers(
  SUBDIR     viame
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_image_processing_export.h
  NOPATH   SUBDIR     viame
  )

kwiver_add_library( viame_image_processing
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_image_processing PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

target_link_libraries( viame_image_processing
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
                       viame_image_ops
                       kwiver::kwiversys
  )

set_target_properties( viame_image_processing PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

algorithms_create_plugin( viame_image_processing
  register_algorithms.cxx
  )

target_link_libraries( viame_image_processing_plugin
  PUBLIC               kwiver::vital_vpm
  )

###
# Process library
##
set( process_sources
  register_processes.cxx
  format_images_srm_process.cxx
)

set( process_headers
  format_images_srm_process.h
)

kwiver_add_plugin( viame_processes_image_processing
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

target_include_directories( viame_processes_image_processing PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
