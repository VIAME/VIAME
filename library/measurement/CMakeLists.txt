# Stereo measurement
#
# Imported from kwiver in P5-T04: arrows/mvg's triangulation and epipolar
# geometry, arrows/ocv's camera intrinsics conversion, and the stereo depth
# map process from sprokit/processes/core. The code is unchanged apart from
# its include paths and its export macro, which is this library's now; the
# namespace is still kwiver's, and phase 8 is where that changes.
#
# VIAME's own measurement code -- the stereo calibration and rectification,
# the detection pairing and the length measurement -- is still in plugins/
# and joins this in phase 2.

set( plugin_headers
  camera_intrinsics.h
  epipolar_geometry.h
  triangulate.h
  )

set( plugin_sources
  camera_intrinsics.cxx
  epipolar_geometry.cxx
  triangulate.cxx
  )

kwiver_install_headers(
  SUBDIR     viame/measurement
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_measurement_export.h
  NOPATH   SUBDIR     viame/measurement
  )

# Address MSVC fatal error C1128:
# number of sections exceeded object file format limit
if( MSVC )
  set_source_files_properties(
    triangulate.cxx
    PROPERTIES COMPILE_FLAGS "/bigobj"
    )
endif()

kwiver_add_library( viame_measurement
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_measurement PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  # OpenCV's headers, for camera_intrinsics
  $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>
  )

target_link_libraries( viame_measurement
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
                       kwiver::kwiversys
  PRIVATE              ${OpenCV_LIBS}
  )

set_target_properties( viame_measurement PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

###
# Processes
##
set( process_headers
  compute_stereo_depth_map_process.h
  )

set( process_sources
  register_processes.cxx
  compute_stereo_depth_map_process.cxx
  )

kwiver_add_plugin( viame_processes_measurement
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
                   viame_measurement
  )

target_include_directories( viame_processes_measurement PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

###
# Python implementations
#
# `ocv_stereo_disparity` was `plugins/opencv/compute_stereo_disparity.cxx`
# until P7-T06: block matching, the WLS filter and `stereoRectify` are
# calib3d and ximgproc rather than image_ops primitives, so the algorithm
# stays cv2's and only the language changes.
##
if( VIAME_ENABLE_PYTHON )
  kwiver_create_python_init( measurement )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    measurement
    __init__ )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/ocv_stereo_disparity.py
    measurement
    ocv_stereo_disparity )

  # `ocv_detect_calibration_targets`, and the four detection helpers it used
  # from `calibrate_stereo_cameras`.
  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/ocv_calibration_targets.py
    measurement
    ocv_calibration_targets )
endif()
