# Object trackers
#
# Imported from kwiver in P5-T04: arrows/core's two thresholding track
# algorithms, and the five track processes from sprokit/processes/core. The
# code is unchanged apart from its include paths and its export macro, which
# is this library's now; the namespace is still kwiver's, and phase 8 is
# where that changes.
#
# VIAME's own trackers -- the pytorch ones, the IOU tracker, the track
# conductor -- are still in plugins/ and join this in phase 2.

set( plugin_headers
  associate_detections_to_tracks_threshold.h
  initialize_object_tracks_threshold.h
  )

set( plugin_sources
  associate_detections_to_tracks_threshold.cxx
  initialize_object_tracks_threshold.cxx
  )

kwiver_install_headers(
  SUBDIR     viame/object_trackers
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_object_trackers_export.h
  NOPATH   SUBDIR     viame/object_trackers
  )

kwiver_add_library( viame_object_trackers
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_object_trackers PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

target_link_libraries( viame_object_trackers
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
                       kwiver::kwiversys
  )

set_target_properties( viame_object_trackers PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

algorithms_create_plugin( viame_object_trackers
  register_algorithms.cxx
  )

target_link_libraries( viame_object_trackers_plugin
  PUBLIC               kwiver::vital_vpm
  )

###
# Processes
##
set( process_headers
  convert_tracks_to_detections_process.h
  initialize_object_tracks_process.h
  merge_track_sets_process.h
  track_objects_process.h
  unwrap_detections_process.h
  )

set( process_sources
  register_processes.cxx
  convert_tracks_to_detections_process.cxx
  initialize_object_tracks_process.cxx
  merge_track_sets_process.cxx
  track_objects_process.cxx
  unwrap_detections_process.cxx
  )

kwiver_add_plugin( viame_processes_object_trackers
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
                   viame_object_trackers
  )

target_include_directories( viame_processes_object_trackers PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
