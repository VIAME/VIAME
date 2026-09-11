# In-house video and image reading
#
# The image_io and video_input implementations VIAME registers. Decoding is
# still delegated to a nested implementation; what lives here is the depth,
# range and frame selection handling the pipelines and tooling rely on.
# Nothing here may include VXL.

# Imported from kwiver in P5-T04: arrows/core's image-list video reader and
# arrows/ocv's image_io. The code is unchanged apart from its include paths
# and its export macro, which is this library's now; the namespace is still
# kwiver's, and phase 8 is where that changes.
set( plugin_headers
  core_image_io.h
  video_input_image_list.h
  )

set( codec_headers
  codecs/image_codec.h
  codecs/tiff.h
  )

set( plugin_sources
  core_image_io.cxx
  video_input_image_list.cxx
  codecs/image_codec.cxx
  codecs/tiff.cxx
  )

kwiver_install_headers(
  SUBDIR     viame/video_io
  ${plugin_headers}
  )

kwiver_install_headers(
  SUBDIR     viame/video_io/codecs
  NOPATH
  ${codec_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_video_io_export.h
  NOPATH   SUBDIR     viame/video_io
  )

kwiver_add_library( viame_video_io
  ${plugin_headers}
  ${codec_headers}
  ${plugin_sources}
  )

target_include_directories( viame_video_io PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

# The vendored stb headers, which only `codecs/image_codec.cxx` compiles.
# SYSTEM: they are third-party code and warn under VIAME's flags.
target_include_directories( viame_video_io SYSTEM PRIVATE
  $<BUILD_INTERFACE:${VIAME_SOURCE_DIR}/third_party/stb>
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
# Processes
#
# Imported from sprokit/processes in P5-T04: the five video and image
# processes from `core`. `image_viewer` came from `ocv` and is python now --
# it is the one process that opens a GUI, which is highgui and has no
# `image_ops` counterpart.
##
set( process_headers
  frame_list_process.h
  image_file_reader_process.h
  image_writer_process.h
  video_input_process.h
  video_output_process.h
  )

set( process_sources
  register_processes.cxx
  frame_list_process.cxx
  image_file_reader_process.cxx
  image_writer_process.cxx
  video_input_process.cxx
  video_output_process.cxx
  )

kwiver_add_plugin( viame_processes_video_io
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
                   viame_video_io
  )

target_include_directories( viame_processes_video_io PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

###
# Python implementations
##
if( VIAME_ENABLE_PYTHON )
  kwiver_create_python_init( video_io )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    video_io
    __init__ )

  # `image_viewer`, a sprokit process rather than an algorithm, so it is
  # registered through `__sprokit_register__`.
  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/image_viewer.py
    video_io
    image_viewer )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/pyav_video_input.py
    video_io
    pyav_video_input )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/pyav_video_output.py
    video_io
    pyav_video_output )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg_cli_video_input.py
    video_io
    ffmpeg_cli_video_input )

  # The image_io the in-house codecs fall back to; see codecs/image_codec.h
  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/pil_image_io.py
    video_io
    pil_image_io )
endif()
