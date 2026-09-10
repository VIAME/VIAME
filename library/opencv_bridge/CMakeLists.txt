# The OpenCV bridge
#
# The cv::Mat-backed implementations of the core types -- image_container,
# mat_image_memory, descriptor_set, feature_set, match_set and the bounding
# box conversion -- imported from arrows/ocv in P5-T04.
#
# This directory is transitional and phase 7 deletes it. It is not one of
# the libraries in design/lite-library-layout.md's target tree: the bridge
# exists only so that the OpenCV-using code can keep converting to and from
# `vital::image` while it is being ported, and P7-T04's "remove
# arrows/ocv/image_container.h uses" is what removes its last caller. It sits
# where image_ops sits in the dependency graph, below video_io and
# image_processing, but it is kept apart from image_ops on purpose: image_ops
# is the code that replaces OpenCV, so it must not link it.

set( plugin_headers
  bounding_box.h
  descriptor_set.h
  feature_set.h
  image_container.h
  mat_image_memory.h
  match_set.h
  matrix.h
  )

set( plugin_sources
  descriptor_set.cxx
  feature_set.cxx
  image_container.cxx
  mat_image_memory.cxx
  match_set.cxx
  )

kwiver_install_headers(
  SUBDIR     viame/opencv_bridge
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_opencv_bridge_export.h
  NOPATH   SUBDIR     viame/opencv_bridge
  )

kwiver_add_library( viame_opencv_bridge
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_opencv_bridge PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>
  )

target_link_libraries( viame_opencv_bridge
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
                       kwiver::kwiversys
                       ${OpenCV_LIBS}
  )

# `mat_image_memory` and `image_container` still carry OpenCV 2 branches on
# cv::Mat::refcount, so the version has to reach them. PUBLIC because the
# guard is in a header every consumer includes.
target_compile_definitions( viame_opencv_bridge
  PUBLIC KWIVER_OPENCV_VERSION_MAJOR=${KWIVER_OPENCV_VERSION_MAJOR}
  )

set_target_properties( viame_opencv_bridge PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )
