# Classifiers and detection filters
#
# Imported from kwiver in P5-T04: arrows/core's class probability filter, and
# the four refinement and filtering processes from sprokit/processes/core.
# The code is unchanged apart from its include paths and its export macro,
# which is this library's now; the namespace is still kwiver's, and phase 8
# is where that changes.
#
# VIAME's own refiners and mergers came from the `core` plugin in P2-T05, the
# windowed refiner from merging core's and opencv's, and the ONNX and pytorch
# classifiers and refiners from the `onnx` plugin and the `pytorch` plugin as one
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
# SVM
#
# The SVM refiner and detector trainer, and the process that trains the
# models, from `svm` in P2-T08. Their own library and plugin rather than part
# of `viame_classifiers`, as darknet's detector is: libsvm is optional, and
# what links `viame_classifiers` should not load it. The query process that
# uses the models is `descriptors`'.
##
if( VIAME_ENABLE_SVM )
  viame_install_headers(
    SUBDIR     viame/classifiers
    refine_detections_svm.h
    train_detector_svm.h
    )

  viame_install_headers(
    ${CMAKE_CURRENT_BINARY_DIR}/viame_classifiers_svm_export.h
    NOPATH   SUBDIR     viame/classifiers
    )

  viame_add_library( viame_classifiers_svm
    refine_detections_svm.h
    refine_detections_svm.cxx
    train_detector_svm.h
    train_detector_svm.cxx
    )

  target_include_directories( viame_classifiers_svm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    )

  target_link_libraries( viame_classifiers_svm
    PUBLIC               kwiver::vital
                         kwiver::vital_algo
    PRIVATE              kwiver::vital_config
                         $<BUILD_INTERFACE:viame::libsvm>
    )

  set_target_properties( viame_classifiers_svm PROPERTIES
    SOVERSION            ${VIAME_VERSION_MAJOR}
    )

  viame_register_statically( viame_classifiers_svm
    register_svm.cxx
    )

  viame_add_plugin( viame_processes_classifiers_svm
    KIND        PROCESSES
    SOURCES          register_processes_svm.cxx
                     train_svm_models_process.cxx
                     train_svm_models_process.h
    PRIVATE          kwiver::vital
                     kwiver::vital_algo
                     kwiver::vital_config
                     kwiver::vital_logger
                     kwiver::vital_util
                     kwiver::vital_vpm
                     kwiver::sprokit_pipeline
                     $<BUILD_INTERFACE:viame::libsvm>
    )

  target_include_directories( viame_processes_classifiers_svm PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    )
endif()

###
# Python
#
# The detection mergers and the multi-camera suppressor, from
# the `core` plugin in P2-T05.
##
if( VIAME_ENABLE_PYTHON )
  viame_add_python_package( classifiers
    EXCLUDE "^(onnx|netharn|rf_detr|sleap)/" )

  # The onnxruntime classifier and reclassifier, from the `onnx` plugin in
  # P2-T05, gated the way that directory was.
  viame_add_python_package( classifiers/onnx
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/onnx
    CONDITION VIAME_ENABLE_ONNX )

  # The pytorch classifiers and refiners, from the `pytorch` plugin in P2-T05,
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

###
# Tests
##
if( VIAME_ENABLE_TESTS )
  add_subdirectory( tests )
endif()
