###
# Training
#
# The adaptive detector and tracker trainers, which choose among nested
# trainers by what the data looks like; the training-data utilities the
# `viame train` applet and the trainers share; and the supervisor that
# watches a training run. From the `core` plugin and the `claude` plugin in
# P2-T07, with the python trainers from `viame.core`: ByteTrack and OC-SORT
# parameter estimation, the three-frame difference settings estimation,
# tracker parameter search and the training data handling.
#
# The windowed trainer is the merge of core's and opencv's: one
# implementation registered as `windowed` with `ocv_windowed` an alias, whose
# evidence is the `training` golden group.
##

set( plugin_headers
  adaptive_detector_trainer.h
  adaptive_tracker_trainer.h
  train_supervisor.h
  utilities_training.h
  windowed_trainer.h
  )

set( plugin_sources
  adaptive_detector_trainer.cxx
  adaptive_tracker_trainer.cxx
  train_supervisor.cxx
  utilities_training.cxx
  windowed_trainer.cxx
  )

viame_install_headers(
  SUBDIR     viame/training
  ${plugin_headers}
  )

viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_training_export.h
  NOPATH   SUBDIR     viame/training
  )

viame_add_library( viame_training
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_training PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

target_link_libraries( viame_training
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
                       kwiver::sprokit_pipeline
                       kwiver::kwiver_adapter
                       viame_utilities
                       # the chipping, which the windowed trainer shares with
                       # the windowed detector and refiner
                       viame_image_ops
  )

# The supervisor watches the training process from a thread of its own.
find_package( Threads REQUIRED )
target_link_libraries( viame_training PRIVATE Threads::Threads )

set_target_properties( viame_training PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

viame_register_statically( viame_training
  register_algorithms.cxx
  )

###
# Python
##
if( VIAME_ENABLE_PYTHON )
  viame_add_python_package( training EXCLUDE "^export/" )

  # ONNX export: the `convert_to_onnx` process, which dispatches to each
  # backend's exporter beside that backend, and the darknet exporter, which
  # waits here for darknet to move in P2-T08.
  set( _export_exclude "^$" )
  if( NOT VIAME_ENABLE_DARKNET )
    set( _export_exclude "^darknet_to_onnx\\.py$" )
  endif()
  viame_add_python_package( training/export
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/export
    EXCLUDE ${_export_exclude}
    CONDITION VIAME_ENABLE_ONNX )
endif()

###
# Tests
##
if( VIAME_ENABLE_TESTS )
  add_subdirectory( tests )
endif()
