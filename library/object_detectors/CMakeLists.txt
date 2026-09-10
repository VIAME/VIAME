# Object detectors
#
# Imported from kwiver in P5-T04: arrows/ocv's three detectors, and the two
# detector processes from sprokit/processes/core. The code is unchanged apart
# from its include paths and its export macro, which is this library's now;
# the namespace is still kwiver's, and phase 8 is where that changes.
#
# VIAME's own detectors -- the pytorch ones, the windowed chipper, the empty
# and full-frame detectors -- are still in plugins/ and join this in phase 2.

set( plugin_headers
  detect_heat_map.h
  detect_motion_3frame_differencing.h
  hough_circle_detector.h
  )

set( plugin_sources
  detect_heat_map.cxx
  detect_motion_3frame_differencing.cxx
  hough_circle_detector.cxx
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
  # OpenCV's headers, for the classes imported from arrows/ocv
  $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>
  )

target_link_libraries( viame_object_detectors
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
                       kwiver::kwiversys
  PRIVATE              ${OpenCV_LIBS}
                       kwiver::kwiver_algo_ocv
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

target_include_directories( viame_processes_object_detectors PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  # OpenCV's headers, for the classes imported from arrows/ocv
  $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>
  )
