# In-house image kernels
#
# Templated over the pixel type and header-only apart from two helpers, so a
# consumer mostly includes what it needs and takes vital::image with it.
# Nothing here may include OpenCV or VXL.
#
# The two compiled helpers came from the `core` plugin in P2-T04.
# `convert_polygons_to_mask` and `utilities_segmentation` are the
# polygon-and-mask geometry the viame_csv and dive readers and writers run,
# and lite-library-layout.md's rule is that a library needing a sibling's
# helper moves the helper down. They both build on the kernels here --
# `polygon.h` and `contours.h` -- so this is as far down as they go, and it
# is why the target is a library rather than an INTERFACE.
#
# `windowed_utils` came down here for the same reason in P2-T05, once the
# two chippers became one: the detector is `object_detectors`, the refiner
# is `classifiers` and the trainer is `training`, and it needs nothing above
# `warp.h`.

set( image_ops_headers
  channels.h
  color.h
  commonality.h
  contours.h
  convert.h
  dispatch.h
  draw.h
  filter.h
  font_5x7.h
  histogram.h
  homography_overlap.h
  layout.h
  match.h
  morphology.h
  pixel.h
  polygon.h
  resample.h
  statistics.h
  stretch.h
  temporal.h
  threshold.h
  warp.h

  convert_polygons_to_mask.h
  utilities_segmentation.h
  windowed_utils.h
)

set( image_ops_sources
  convert_polygons_to_mask.cxx
  utilities_segmentation.cxx
  windowed_utils.cxx
)

viame_add_library( viame_image_ops
  ${image_ops_headers}
  ${image_ops_sources}
)
add_library( viame::image_ops ALIAS viame_image_ops )

# The parent directory, so consumers include <image_ops/convert.h> rather than
# <convert.h>: several kernels share a name with the algorithm that uses them
target_include_directories( viame_image_ops PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include/viame>
)

target_link_libraries( viame_image_ops
  PUBLIC  kwiver::vital
          # `windowed_utils` names `image_object_detector` in its interface
          kwiver::vital_algo
          kwiver::vital_config
          kwiver::vital_logger
          kwiver::vital_util
)

set_target_properties( viame_image_ops PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
)

install( FILES       ${image_ops_headers}
         DESTINATION include/viame/image_ops )

install( FILES       ${CMAKE_CURRENT_BINARY_DIR}/viame_image_ops_export.h
         DESTINATION include/viame/image_ops )

###
# Tests
##
if( VIAME_ENABLE_TESTS )
  add_subdirectory( tests )
endif()
