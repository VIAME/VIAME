# Object trackers
#
# Imported from kwiver in P5-T04: arrows/core's two thresholding track
# algorithms, and the five track processes from sprokit/processes/core. The
# code is unchanged apart from its include paths and its export macro, which
# is this library's now; the namespace is still kwiver's, and phase 8 is
# where that changes.
#
# VIAME's own track processes -- accumulation, filtering, resampling, the
# feature-landmark split and the track conductor -- and its python trackers
# (ByteTrack, OC-SORT, the homography IOU trackers, tube-IoU fusion) came from
# `plugins/core` in P2-T06. The pytorch trackers are still in
# `plugins/pytorch`.

set( plugin_headers
  associate_detections_to_tracks_threshold.h
  initialize_object_tracks_threshold.h
  )

set( plugin_sources
  associate_detections_to_tracks_threshold.cxx
  initialize_object_tracks_threshold.cxx
  )

viame_install_headers(
  SUBDIR     viame/object_trackers
  ${plugin_headers}
  )

viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_object_trackers_export.h
  NOPATH   SUBDIR     viame/object_trackers
  )

viame_add_library( viame_object_trackers
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
  )

set_target_properties( viame_object_trackers PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

viame_register_statically( viame_object_trackers
  register_algorithms.cxx
  )


###
# Processes
##
set( process_headers
  accumulate_object_tracks_process.h
  convert_tracks_to_detections_process.h
  filter_object_tracks_process.h
  initialize_object_tracks_process.h
  merge_track_sets_process.h
  resample_object_tracks_process.h
  split_tracks_to_feature_landmarks_process.h
  track_conductor_process.h
  track_objects_process.h
  unwrap_detections_process.h
  )

set( process_sources
  register_processes.cxx
  accumulate_object_tracks_process.cxx
  convert_tracks_to_detections_process.cxx
  filter_object_tracks_process.cxx
  initialize_object_tracks_process.cxx
  merge_track_sets_process.cxx
  resample_object_tracks_process.cxx
  split_tracks_to_feature_landmarks_process.cxx
  track_conductor_process.cxx
  track_objects_process.cxx
  unwrap_detections_process.cxx
  )

viame_add_plugin( viame_processes_object_trackers
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
                   # `filter_object_tracks` turns notes into attributes the
                   # way the CSV readers do
                   viame_file_io
                   viame_object_trackers
  )

target_include_directories( viame_processes_object_trackers PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

###
# Python
#
# ByteTrack and OC-SORT, the simple and multi-camera homography IOU trackers,
# and tube-IoU track-set fusion, from `plugins/core` in P2-T06.
##
if( VIAME_ENABLE_PYTHON )
  viame_add_python_package( object_trackers )
endif()
