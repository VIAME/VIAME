# Descriptor formatting, computation and querying
#
# Nothing here may include VXL.
#
# Imported from kwiver in P5-T04: arrows/core's descriptor request handler,
# and the three descriptor and query processes from sprokit/processes/core.
# The imported code is unchanged apart from its include paths and its export
# macro, which is this library's now; its namespace is still kwiver's, and
# phase 8 is where that changes.
#
# VIAME's own descriptor processes -- ingest, fetch, per-track descriptors,
# database query creation and selection, writing results as tracks, the IQR
# AdaBoost query -- came from the `core` plugin in P2-T06, with the track
# descriptor averaging, the CSV descriptor query and the header-only IQR
# session. The torchvision extractors came from the `pytorch` plugin as a gated
# python subpackage.

set( plugin_headers
  average_track_descriptors.h
  handle_descriptor_request_core.h
  iqr_session.h
  query_track_descriptor_set_csv.h
  utilities_iqr.h
  )

set( plugin_sources
  average_track_descriptors.cxx
  handle_descriptor_request_core.cxx
  query_track_descriptor_set_csv.cxx
  )

viame_install_headers(
  SUBDIR     viame/descriptors
  ${plugin_headers}
  )

viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_descriptors_export.h
  NOPATH   SUBDIR     viame/descriptors
  )

viame_add_library( viame_descriptors
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_descriptors PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

target_link_libraries( viame_descriptors
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
  )

set_target_properties( viame_descriptors PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

viame_register_statically( viame_descriptors
  register_algorithms.cxx
  )


###
# The IQR AdaBoost session
#
# Its own library, and the only thing in VIAME that calls the interpreter
# from C++: it trains and scores through `viame.descriptors.iqr_adaboost`,
# which is scikit-learn. Kept out of `viame_descriptors` on purpose -- the
# libraries everything links stay free of libpython, and only what needs the
# model picks the dependency up. From the `core` plugin in P2-T06, under the
# same target name.
#
# Without python it compiles to a stub that reports no model, and the base
# class falls back to similarity scoring against the positive exemplars, so
# the query still returns ranked results.
##
viame_install_headers(
  SUBDIR     viame/descriptors
  iqr_session_adaboost.h
  )

viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_iqr_export.h
  NOPATH   SUBDIR     viame/descriptors
  )

viame_add_library( viame_iqr
  iqr_session_adaboost.h
  iqr_session_adaboost.cxx
  )

target_include_directories( viame_iqr PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

target_link_libraries( viame_iqr
  PUBLIC               kwiver::vital
                       kwiver::vital_logger
  )

set_target_properties( viame_iqr PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

if( VIAME_ENABLE_PYTHON )
  target_compile_definitions( viame_iqr PRIVATE VIAME_CORE_HAVE_PYTHON )

  # `${PYTHON_LIBRARIES}` as well as pybind11: VIAME links with
  # `-Wl,--no-undefined`, so leaving the interpreter symbols to be resolved
  # at load time does not link at all here.
  target_link_libraries( viame_iqr
    PRIVATE ${PYTHON_LIBRARIES} $<BUILD_INTERFACE:pybind11::pybind11> )
endif()


###
# Processes
##
set( process_headers
  compute_track_descriptors_process.h
  create_database_query_process.h
  extract_desc_ids_for_training_process.h
  fetch_descriptors_process.h
  handle_descriptor_request_process.h
  ingest_descriptors_process.h
  object_track_descriptors_process.h
  perform_query_process.h
  process_query_process_adaboost.h
  select_database_query_process.h
  write_query_results_as_tracks_process.h
)

set( process_sources
  register_processes.cxx
  compute_track_descriptors_process.cxx
  create_database_query_process.cxx
  extract_desc_ids_for_training_process.cxx
  fetch_descriptors_process.cxx
  handle_descriptor_request_process.cxx
  ingest_descriptors_process.cxx
  object_track_descriptors_process.cxx
  perform_query_process.cxx
  process_query_process_adaboost.cxx
  select_database_query_process.cxx
  write_query_results_as_tracks_process.cxx
)

viame_add_plugin( viame_processes_descriptors
  KIND        PROCESSES
  SOURCES          ${process_sources}
                   ${process_headers}
  PRIVATE          kwiver::sprokit_pipeline
                   kwiver::kwiver_adapter
                   kwiver::vital
                   kwiver::vital_vpm
                   kwiver::vital_logger
                   kwiver::vital_config
                   kwiver::vital_util
                   kwiver::vital_exceptions
                   kwiver::vital_algo
                   viame_image_ops
                   viame_descriptors
                   # `process_query_adaboost` ranks through the IQR session
                   viame_iqr
 )

target_include_directories( viame_processes_descriptors PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

###
# SVM query
#
# The IQR session over SVM models and the process that runs a query through
# it, from `svm` in P2-T08, as their own plugin because libsvm is optional.
# The models come from `classifiers`' `train_svm_models`.
##
if( VIAME_ENABLE_SVM )
  viame_add_plugin( viame_processes_descriptors_svm
    KIND        PROCESSES
    SOURCES          register_processes_svm.cxx
                     process_query_process.cxx
                     process_query_process.h
                     iqr_session_svm.h
    PRIVATE          kwiver::vital
                     kwiver::vital_algo
                     kwiver::vital_config
                     kwiver::vital_logger
                     kwiver::vital_util
                     kwiver::vital_vpm
                     kwiver::sprokit_pipeline
                     viame_descriptors
                     $<BUILD_INTERFACE:viame::libsvm>
    )

  target_include_directories( viame_processes_descriptors_svm PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    )
endif()

###
# Python
#
# The IQR model and the indexing and query tools from the `core` plugin, and the
# torchvision extractors from the `pytorch` plugin as a subpackage gated on
# their option.
##
if( VIAME_ENABLE_PYTHON )
  viame_add_python_package( descriptors EXCLUDE "^torchvision/" )

  viame_add_python_package( descriptors/torchvision
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/torchvision
    CONDITION VIAME_ENABLE_PYTORCH VIAME_ENABLE_PYTORCH-VISION )
endif()
