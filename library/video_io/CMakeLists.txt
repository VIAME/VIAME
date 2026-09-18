# In-house video reading and writing
#
# The video_input implementations VIAME registers, and the processes that
# drive a stream. Decoding is still delegated to a nested implementation;
# what lives here is the range and frame selection handling the pipelines
# and tooling rely on. Nothing here may include VXL.
#
# The image side is `image_io`: one image decoded, named or written. What
# stayed is about a sequence. `video_input_image_list` is the boundary case
# and stays here, because it registers as a `video_input` under the name
# `image_list` -- every frame it yields comes through an `image_io`, but the
# thing it implements is a video source.

# Imported from kwiver in P5-T04: arrows/core's image-list video reader. The
# code is unchanged apart from its include paths and its export macro, which
# is this library's now; the namespace is still kwiver's, and phase 8 is
# where that changes.
#
# `detect_shot_breaks` came from the `core` plugin in P2-T04: where one shot
# ends is a property of the video, not of whatever happened to read it.
set( plugin_headers
  detect_shot_breaks.h
  video_input_image_list.h
  )

set( plugin_sources
  detect_shot_breaks.cxx
  video_input_image_list.cxx
  )

viame_install_headers(
  SUBDIR     viame/video_io
  ${plugin_headers}
  )

viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_video_io_export.h
  NOPATH   SUBDIR     viame/video_io
  )

viame_add_library( viame_video_io
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
                       viame_image_kernels
  )

set_target_properties( viame_video_io PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

viame_register_statically( viame_video_io
  register_algorithms.cxx
  )


###
# Processes
#
# Imported from sprokit/processes in P5-T04: the video processes from
# `core`. The two single-image processes went to `image_io`.
##
set( process_headers
  detect_shot_breaks_process.h
  frame_list_process.h
  read_habcam_metadata_process.h
  video_input_process.h
  video_output_process.h
  )

set( process_sources
  register_processes.cxx
  detect_shot_breaks_process.cxx
  frame_list_process.cxx
  read_habcam_metadata_process.cxx
  video_input_process.cxx
  video_output_process.cxx
  )

viame_add_plugin( viame_processes_video_io
  KIND        PROCESSES
  SOURCES          ${process_sources}
                   ${process_headers}
  PRIVATE          kwiver::sprokit_pipeline
                   kwiver::vital
                   kwiver::vital_vpm
                   kwiver::vital_logger
                   kwiver::vital_config
                   kwiver::vital_util
                   kwiver::vital_exceptions
                   kwiver::vital_algo
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
  viame_create_python_init( video_io )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    video_io
    __init__ )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/pyav_video_input.py
    video_io
    pyav_video_input )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/pyav_video_output.py
    video_io
    pyav_video_output )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg_cli_video_input.py
    video_io
    ffmpeg_cli_video_input )
endif()

###
# Tests
##
if( VIAME_ENABLE_TESTS )
  add_subdirectory( tests )
endif()
