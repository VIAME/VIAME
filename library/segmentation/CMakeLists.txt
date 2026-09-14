###
# Segmentation
#
# Setting a detection's mask, and the interactive service that does it from
# a few clicks. `add_keypoints_from_mask` -- which measures a mask and hangs
# keypoints off the detection -- came from the `opencv` plugin in P2-T05; it
# has not included OpenCV since P7-T04b, so the directory it was in was
# named for a dependency it no longer had.
#
# The polygon-and-mask geometry the readers and writers share went to
# `image_ops` in P2-T04, one level down, because `file_io`'s CSV readers
# need it too. The SAM family came from the `pytorch` plugin in P2-T05, as
# `sam2/` and `sam3/` subpackages gated on their options.
##

set( plugin_headers
  add_keypoints_from_mask.h
  )

set( plugin_sources
  add_keypoints_from_mask.cxx
  )

viame_install_headers(
  SUBDIR     viame/segmentation
  ${plugin_headers}
  )

viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_segmentation_export.h
  NOPATH   SUBDIR     viame/segmentation
  )

viame_add_library( viame_segmentation
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_segmentation PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

target_link_libraries( viame_segmentation
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
                       # contour tracing and the morphology the mask
                       # measurement runs
                       viame_image_ops
  )

set_target_properties( viame_segmentation PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

viame_register_statically( viame_segmentation
  register_algorithms.cxx
  )

###
# Python
##
if( VIAME_ENABLE_PYTHON )
  viame_add_python_package( segmentation EXCLUDE "^(sam2|sam3)/" )

  # The SAM family, from the `pytorch` plugin in P2-T05, one subpackage per
  # option.
  foreach( gate IN ITEMS "sam2:SAM2" "sam3:SAM3" )
    string( REPLACE ":" ";" gate "${gate}" )
    list( GET gate 0 subdir )
    list( GET gate 1 option )
    viame_add_python_package( segmentation/${subdir}
      DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}
      CONDITION VIAME_ENABLE_PYTORCH VIAME_ENABLE_PYTORCH-${option} )
  endforeach()
endif()
