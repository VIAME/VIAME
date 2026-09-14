# Stereo measurement
#
# Imported from kwiver in P5-T04: arrows/mvg's triangulation and epipolar
# geometry, arrows/ocv's camera intrinsics conversion, and the stereo depth
# map process from sprokit/processes/core. The code is unchanged apart from
# its include paths and its export macro, which is this library's now; the
# namespace is still kwiver's, and phase 8 is where that changes.
#
# VIAME's own measurement code came from `plugins/core` and `plugins/opencv`
# in P2-T06: the stereo measurement utilities, detection and track pairing,
# and the measurement, refinement and calibration-from-tracks processes. The
# task text had the core and opencv pairing and measurement merged into one
# implementation each; they no longer share a config surface -- see
# design/STATUS.md P2-T06 -- so both move as themselves, under their own
# registered names, the opencv files with an `ocv_` prefix.

set( plugin_headers
  camera_intrinsics.h
  epipolar_geometry.h
  measurement_utilities.h
  ocv_pair_stereo_detections.h
  ocv_pair_stereo_tracks.h
  pair_stereo_detections.h
  pair_stereo_tracks.h
  projection.h
  triangulate.h
  )

set( plugin_sources
  camera_intrinsics.cxx
  epipolar_geometry.cxx
  measurement_utilities.cxx
  ocv_pair_stereo_detections.cxx
  ocv_pair_stereo_tracks.cxx
  pair_stereo_detections.cxx
  pair_stereo_tracks.cxx
  projection.cxx
  triangulate.cxx
  )

viame_install_headers(
  SUBDIR     viame/measurement
  ${plugin_headers}
  )

viame_install_headers(
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

viame_add_library( viame_measurement
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_measurement PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

target_link_libraries( viame_measurement
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
                       # the calibration and track readers
                       viame_file_io
                       # classification averaging over a stereo pair
                       viame_utilities
                       # the chipping the pairing crops through
                       viame_image_ops
                       # the debug images the pairing can write
                       viame_video_io
  )

# The pairing's DINO feature matching calls the interpreter through the
# Python C API, which is why `viame_core` carried libpython until this code
# left it.
if( VIAME_ENABLE_PYTHON )
  target_include_directories( viame_measurement SYSTEM PRIVATE ${PYTHON_INCLUDE_DIRS} )
  target_compile_definitions( viame_measurement PUBLIC VIAME_ENABLE_PYTHON )
  target_link_libraries( viame_measurement PRIVATE ${PYTHON_LIBRARIES} )
endif()

set_target_properties( viame_measurement PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

###
# Processes
##
set( process_headers
  calibrate_cameras_from_tracks_process.h
  compute_stereo_depth_map_process.h
  measure_objects_process.h
  ocv_measure_objects_process.h
  ocv_pair_stereo_detections_process.h
  pair_stereo_detections_process.h
  pair_stereo_tracks_process.h
  refine_measurements_process.h
  )

set( process_sources
  register_processes.cxx
  calibrate_cameras_from_tracks_process.cxx
  compute_stereo_depth_map_process.cxx
  measure_objects_process.cxx
  ocv_measure_objects_process.cxx
  ocv_pair_stereo_detections_process.cxx
  pair_stereo_detections_process.cxx
  pair_stereo_tracks_process.cxx
  refine_measurements_process.cxx
  )

viame_add_plugin( viame_processes_measurement
  KIND        PROCESSES
  SOURCES          ${process_sources}
                   ${process_headers}
  PRIVATE          kwiver::sprokit_pipeline
                   kwiver::vital
                   kwiver::vital_vpm
                   kwiver::vital_logger
                   kwiver::vital_config
                   kwiver::vital_util
                   kwiver::vital_exceptions
                   kwiver::vital_algo
                   kwiver::kwiver_adapter
                   viame_measurement
                   # `measure_using_stereo` hangs keypoints off a mask. A
                   # sibling link, carried over from `viame_processes_opencv`
                   # rather than introduced; the helper moving down is the
                   # document's rule and a later step's job.
                   viame_segmentation
  )

target_include_directories( viame_processes_measurement PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

###
# Python
#
# Everything in the directory: the OpenCV replacements from P7-T06
# (`ocv_stereo_disparity`, the calibration targets and optimisation,
# `stereo_frame_selection`, the single-camera calibration process), and from
# P2-T06 the interactive stereo service, curved measurement, survey metadata,
# and opencv's stereo utilities, pipeline and prior coverage.
##
if( VIAME_ENABLE_PYTHON )
  viame_add_python_package( measurement )

  # Compiled bindings exposing the C++ stereo measurement to python, so the
  # interactive stereo service reuses it rather than duplicating the maths.
  # They live at the bottom of `measurement_utilities.cxx` behind
  # `VIAME_MEASUREMENT_PYTHON_BINDINGS`; defining it here makes this build of
  # the file emit only the module. `viame.measurement._measurement` until P2-T06.
  viame_add_python_library( _measurement
    measurement
    SOURCES measurement_utilities.cxx )
  target_compile_definitions( python-measurement-_measurement
    PRIVATE VIAME_MEASUREMENT_PYTHON_BINDINGS )
  target_link_libraries( python-measurement-_measurement
    PRIVATE ${PYTHON_LIBRARIES} $<BUILD_INTERFACE:pybind11::pybind11> viame_measurement )
endif()
