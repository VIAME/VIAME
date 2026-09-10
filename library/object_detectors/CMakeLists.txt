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
  )

set( plugin_sources
  detect_heat_map.cxx
  detect_motion_3frame_differencing.cxx
  example_detector.cxx
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
