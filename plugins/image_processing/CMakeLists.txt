# In-house image processing algorithms
#
# These replace the third-party implementations one at a time, keeping the
# registered names and config keys. Nothing here may include OpenCV or VXL.

set( plugin_headers
  average_frames.h
  color_commonality.h
  convert_image.h
  morphology.h
  threshold.h
  )

set( plugin_sources
  average_frames.cxx
  color_commonality.cxx
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
