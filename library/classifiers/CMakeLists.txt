# Classifiers and detection filters
#
# Imported from kwiver in P5-T04: arrows/core's class probability filter, and
# the four refinement and filtering processes from sprokit/processes/core.
# The code is unchanged apart from its include paths and its export macro,
# which is this library's now; the namespace is still kwiver's, and phase 8
# is where that changes.
#
# VIAME's own refiners and classifiers -- the pytorch and ONNX ones, the
# detection mergers -- are still in plugins/ and join this in phase 2.

set( plugin_headers
  class_probability_filter.h
  )

set( plugin_sources
  class_probability_filter.cxx
  )

kwiver_install_headers(
  SUBDIR     viame/classifiers
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_classifiers_export.h
  NOPATH   SUBDIR     viame/classifiers
  )

kwiver_add_library( viame_classifiers
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_classifiers PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

target_link_libraries( viame_classifiers
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
                       kwiver::kwiversys
  )

set_target_properties( viame_classifiers PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

algorithms_create_plugin( viame_classifiers
  register_algorithms.cxx
  )

target_link_libraries( viame_classifiers_plugin
  PUBLIC               kwiver::vital_vpm
  )

###
# Processes
##
set( process_headers
  detected_object_filter_process.h
  merge_detection_sets_process.h
  refine_detections_process.h
  refine_tracks_process.h
  )

set( process_sources
  register_processes.cxx
  detected_object_filter_process.cxx
  merge_detection_sets_process.cxx
  refine_detections_process.cxx
  refine_tracks_process.cxx
  )

kwiver_add_plugin( viame_processes_classifiers
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
                   viame_classifiers
  )

target_include_directories( viame_processes_classifiers PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
