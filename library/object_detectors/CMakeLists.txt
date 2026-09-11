# Object detectors
#
# Imported from kwiver in P5-T04: arrows/ocv's three detectors, and the two
# detector processes from sprokit/processes/core. The namespace is still
# kwiver's, and phase 8 is where that changes.
#
# P7-T04 took OpenCV out: the two detectors that stayed C++ run on
# `image_ops` and `hough_circle` is `hough_circle_detector.py`, so this
# directory links neither OpenCV nor the bridge.
#
# VIAME's own detectors -- the pytorch ones, the windowed chipper, the empty
# and full-frame detectors -- are still in plugins/ and join this in phase 2.

set( plugin_headers
  detect_heat_map.h
  detect_motion_3frame_differencing.h
  example_detector.h
  windowed_utils.h
  )

set( plugin_sources
  detect_heat_map.cxx
  detect_motion_3frame_differencing.cxx
  example_detector.cxx
  windowed_utils.cxx
  )


kwiver_install_headers(
  SUBDIR     viame/object_detectors
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_object_detectors_export.h
  NOPATH   SUBDIR     viame/object_detectors
  )

kwiver_add_library( viame_object_detectors
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_object_detectors PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

target_link_libraries( viame_object_detectors
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
                       kwiver::kwiversys
                       viame_image_ops
                       # codecs/, for the debug frames the motion detector
                       # saves when a debug_dir is configured
                       viame_video_io
  )


set_target_properties( viame_object_detectors PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

algorithms_create_plugin( viame_object_detectors
  register_algorithms.cxx
  )

target_link_libraries( viame_object_detectors_plugin
  PUBLIC               kwiver::vital_vpm
  )

###
# darknet's detector
#
# `lite-library-layout.md` puts it in this directory, and it was
# `plugins/darknet` until P7-T08's follow-up moved it here. Its trainer went
# to `library/training` at the same time.
#
# **Its own library rather than part of `viame_object_detectors`**, which is
# not tidiness. `libdarknet.so` drags OpenCV in behind it, and
# `viame_object_detectors` is what `plugins/core` links for `windowed_utils`
# -- so folding darknet into it puts libdarknet, and OpenCV, on the load
# path of everything that links `viame_core`. A gtest that touches none of
# it then fails to start for want of `libopencv_highgui`. An optional
# third-party dependency belongs behind its own door.
##
if( VIAME_ENABLE_DARKNET )
  # Add fletch cmake directory for FindCUDNN.cmake needed by DarknetConfig
  list( INSERT CMAKE_MODULE_PATH 0 "${VIAME_PACKAGES_DIR}/fletch/CMake" )

  find_package( Darknet REQUIRED )

  kwiver_install_headers(
    SUBDIR     viame/object_detectors
    darknet_detector.h
    )

  kwiver_install_headers(
    ${CMAKE_CURRENT_BINARY_DIR}/viame_object_detectors_darknet_export.h
    NOPATH   SUBDIR     viame/object_detectors
    )

  kwiver_add_library( viame_object_detectors_darknet
    darknet_detector.h
    darknet_detector.cxx
    )

  target_include_directories( viame_object_detectors_darknet PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    )

  # darknet's headers come from the dependency tree this build takes fletch
  # from; phase 1 vendors them. SYSTEM, so the prefix lands after every -I:
  # VIAME installs its own headers into the same prefix and an -I would let
  # the copy left by the last install shadow the tree being built.
  target_include_directories( viame_object_detectors_darknet SYSTEM PUBLIC
    $<BUILD_INTERFACE:${VIAME_DEPENDENCY_INCLUDE_DIRS}>
    )

  target_link_libraries( viame_object_detectors_darknet
    PUBLIC               kwiver::vital
                         kwiver::vital_algo
                         kwiver::vital_config
                         kwiver::vital_logger
                         viame_object_detectors
    PRIVATE              Darknet::dark
                         viame_image_ops
                         kwiver::kwiversys
    )

  set_target_properties( viame_object_detectors_darknet PROPERTIES
    SOVERSION            ${VIAME_VERSION_MAJOR}
    )

  algorithms_create_plugin( viame_object_detectors_darknet
    register_darknet.cxx
    )

  target_link_libraries( viame_object_detectors_darknet_plugin
    PUBLIC               kwiver::vital_vpm
    )
endif()

###
# Processes
##
set( process_headers
  detect_motion_process.h
  image_object_detector_process.h
  )

set( process_sources
  register_processes.cxx
  detect_motion_process.cxx
  image_object_detector_process.cxx
  )

kwiver_add_plugin( viame_processes_object_detectors
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
                   viame_object_detectors
  )

###
# Python implementations
##
if( VIAME_ENABLE_PYTHON )
  kwiver_create_python_init( object_detectors )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    object_detectors
    __init__ )

  # `hough_circle` was C++ until P7-T04; the Hough gradient transform is not
  # an image_ops primitive, so it stays cv2's and moves to python.
  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/hough_circle_detector.py
    object_detectors
    hough_circle_detector )
endif()

target_include_directories( viame_processes_object_detectors PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
