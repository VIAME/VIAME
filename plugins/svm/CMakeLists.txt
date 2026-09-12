# Build/install plugin for SVM support


# libsvm is vendored in `third_party/libsvm` -- two files, and not stock:
# the `HISTOGRAM` and `NMI` kernels both callers here select are VIAME's own
# and have never existed upstream. It was looked up in the install prefix
# with `NO_DEFAULT_PATH` until P1-T03 brought it into the tree, which is one
# of the couplings that made a second build configuration impossible to
# configure.

set( headers
  refine_detections_svm.h
  train_detector_svm.h )

set( sources
  refine_detections_svm.cxx
  train_detector_svm.cxx )

kwiver_install_headers(
  SUBDIR     viame
  ${headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_svm_export.h
  NOPATH   SUBDIR     viame
  )

kwiver_add_library( viame_svm
  ${headers}
  ${sources}
  )

target_include_directories( viame_svm PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

# `svm.h` comes from `viame::libsvm`, which carries its own include
# directory. It used to come from the install prefix instead -- where
# fletch's copy also sits, so the vendored header was being shadowed by the
# one it replaced.

target_link_libraries( viame_svm
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
  PRIVATE              kwiver::vital_config
                       kwiver::kwiversys
                       $<BUILD_INTERFACE:viame::libsvm>
  )

set_target_properties( viame_svm PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

algorithms_create_plugin( viame_svm
  register_algorithms.cxx
  )


# Add process library
set( process_headers
  process_query_process.h
  train_svm_models_process.h )

set( process_sources
  register_processes.cxx
  process_query_process.cxx
  train_svm_models_process.cxx
)

kwiver_add_plugin( viame_processes_svm
  SUBDIR           ${viame_plugin_process_subdir}
  SOURCES          ${process_sources}
  PRIVATE          kwiver::vital
                   kwiver::vital_algo
                   kwiver::vital_config
                   kwiver::vital_logger
                   kwiver::vital_util
                   kwiver::vital_vpm
                   kwiver::sprokit_pipeline
                   $<BUILD_INTERFACE:viame::libsvm>
 )

target_include_directories( viame_processes_svm PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/plugins>
  )


