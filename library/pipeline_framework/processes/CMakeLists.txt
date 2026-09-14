# Processes that belong to the pipeline framework itself
#
# Imported from kwiver with sprokit in P5-T03; registered here rather than by
# kwiver's `kwiver_processes` plugin since P5-T04. Frame filtering by
# detections and by index, and the image-to-image-set conversion, came from
# the `core` plugin in P2-T07.

set( process_headers
  downsample_process.h
  filter_frame_index_process.h
  filter_frame_process.h
  image_to_image_set_process.h
  )

set( process_sources
  register_processes.cxx
  downsample_process.cxx
  filter_frame_index_process.cxx
  filter_frame_process.cxx
  image_to_image_set_process.cxx
  )

viame_add_plugin( viame_processes_pipeline_framework
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
                   kwiver::kwiver_adapter
  )

target_include_directories( viame_processes_pipeline_framework PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
