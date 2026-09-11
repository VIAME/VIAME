# Detector and tracker trainers
#
# The first inhabitant is darknet's, moved out of `plugins/darknet` in
# P7-T08's follow-up; `lite-library-layout.md` sends the rest here in phase
# 2 -- the adaptive trainers, the windowed trainer, the kwcoco base and the
# python trainers.
#
# Gated on `VIAME_ENABLE_DARKNET`, because everything in it is darknet's so
# far. When phase 2 brings the others across, the library stops being
# optional and only its darknet sources stay behind the flag.

if( NOT VIAME_ENABLE_DARKNET )
  return()
endif()

# Add fletch cmake directory for FindCUDNN.cmake needed by DarknetConfig.cmake
list( INSERT CMAKE_MODULE_PATH 0 "${VIAME_PACKAGES_DIR}/fletch/CMake" )

find_package( Darknet REQUIRED )

set( plugin_headers
  darknet_trainer.h
  )

set( plugin_sources
  darknet_trainer.cxx
  )

kwiver_install_headers(
  SUBDIR     viame/training
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_training_export.h
  NOPATH   SUBDIR     viame/training
  )

kwiver_add_library( viame_training
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_training PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

# darknet's headers, which come from the dependency tree this build takes
# fletch from; phase 1 vendors them. SYSTEM, so the prefix reaches the
# compiler as -isystem and lands after every -I: VIAME installs its own
# headers into the same prefix, and an -I would let the copy left there by
# the last install shadow the tree being built.
target_include_directories( viame_training SYSTEM PUBLIC
  $<BUILD_INTERFACE:${VIAME_DEPENDENCY_INCLUDE_DIRS}>
  )

target_link_libraries( viame_training
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
                       kwiver::kwiversys
                       # `format_image`, the chipping and the crop
                       viame_object_detectors
  PRIVATE              Darknet::dark
                       viame_image_ops
  )

set_target_properties( viame_training PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

algorithms_create_plugin( viame_training
  register_algorithms.cxx
  )

target_link_libraries( viame_training_plugin
  PUBLIC               kwiver::vital_vpm
  )
