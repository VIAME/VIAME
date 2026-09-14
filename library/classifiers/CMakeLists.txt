# Classifiers and detection filters
#
# Imported from kwiver in P5-T04: arrows/core's class probability filter, and
# the four refinement and filtering processes from sprokit/processes/core.
# The code is unchanged apart from its include paths and its export macro,
# which is this library's now; the namespace is still kwiver's, and phase 8
# is where that changes.
#
# VIAME's own refiners and mergers came from `plugins/core` in P2-T05, the
# windowed refiner from merging core's and opencv's, and the ONNX and pytorch
# classifiers and refiners from `plugins/onnx` and `plugins/pytorch` as one
# python subpackage per option.

set( plugin_headers
  class_probability_filter.h
  convert_head_tail_points.h
  merge_detections_suppress_in_regions.h
  refine_detections_add_fixed.h
  refine_detections_nms.h
  refine_tracks_average_tot.h
  windowed_refiner.h
  )

set( plugin_sources
  class_probability_filter.cxx
  convert_head_tail_points.cxx
  merge_detections_suppress_in_regions.cxx
  refine_detections_add_fixed.cxx
  refine_detections_nms.cxx
  refine_tracks_average_tot.cxx
  windowed_refiner.cxx
  )

viame_install_headers(
  SUBDIR     viame/classifiers
  ${plugin_headers}
  )

viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_classifiers_export.h
  NOPATH   SUBDIR     viame/classifiers
  )

viame_add_library( viame_classifiers
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
                       # `refine_tracks_average_tot` averages the
                       # classification scores over a track
                       viame_utilities
                       # `windowed_refiner`'s chipping
                       viame_image_ops
  )

set_target_properties( viame_classifiers PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

viame_register_statically( viame_classifiers
  register_algorithms.cxx
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

viame_add_plugin( viame_processes_classifiers
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
                   viame_classifiers
  )

target_include_directories( viame_processes_classifiers PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

###
# Python
#
# The detection mergers and the multi-camera suppressor, from
# `plugins/core` in P2-T05.
##
if( VIAME_ENABLE_PYTHON )
  viame_add_python_package( classifiers
    EXCLUDE "^(onnx|netharn|rf_detr|sleap)/" )

  # The onnxruntime classifier and reclassifier, from `plugins/onnx` in
  # P2-T05, gated the way that directory was.
  viame_add_python_package( classifiers/onnx
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/onnx
    CONDITION VIAME_ENABLE_ONNX )

  # The pytorch classifiers and refiners, from `plugins/pytorch` in P2-T05,
  # one subpackage per option.
  foreach( gate IN ITEMS "netharn:NETHARN" "rf_detr:RF-DETR" "sleap:SLEAP" )
    string( REPLACE ":" ";" gate "${gate}" )
    list( GET gate 0 subdir )
    list( GET gate 1 option )
    viame_add_python_package( classifiers/${subdir}
      DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}
      CONDITION VIAME_ENABLE_PYTORCH VIAME_ENABLE_PYTORCH-${option} )
  endforeach()
endif()
