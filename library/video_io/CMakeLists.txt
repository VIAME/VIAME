# In-house video and image reading
#
# The image_io and video_input implementations VIAME registers. Decoding is
# still delegated to a nested implementation; what lives here is the depth,
# range and frame selection handling the pipelines and tooling rely on.
# Nothing here may include VXL.

set( plugin_headers
  core_image_io.h
  )

set( plugin_sources
  core_image_io.cxx
  )

kwiver_install_headers(
  SUBDIR     viame
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_video_io_export.h
  NOPATH   SUBDIR     viame
  )

kwiver_add_library( viame_video_io
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_video_io PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

target_link_libraries( viame_video_io
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
                       kwiver::kwiversys
                       viame_image_ops
                       kwiver::kwiver_algo_ffmpeg
  )

set_target_properties( viame_video_io PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

algorithms_create_plugin( viame_video_io
  register_algorithms.cxx
  )

target_link_libraries( viame_video_io_plugin
  PUBLIC               kwiver::vital_vpm
  )

###
# Python implementations
##
if( VIAME_ENABLE_PYTHON )
  kwiver_create_python_init( video_io )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/python/__init__.py
    video_io
    __init__ )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pyav_video_input.py
    video_io
    pyav_video_input )
endif()
