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
# The empty and full-frame detectors came from `plugins/core` in P2-T05,
# and `windowed_detector` is the merge of the two chippers that were in
# `plugins/core` and `plugins/opencv`: one implementation, registered as
# `windowed` with `ocv_windowed` an alias. `windowed_utils` went down to
# `image_ops`, where the refiner and the trainer can reach it too. VIAME's
# own pytorch detectors are still in plugins/.

set( plugin_headers
  detect_heat_map.h
  empty_detector.h
  detect_motion_3frame_differencing.h
  example_detector.h
  full_frame_detector.h
  windowed_detector.h
  )

set( plugin_sources
  detect_heat_map.cxx
  empty_detector.cxx
  detect_motion_3frame_differencing.cxx
  example_detector.cxx
  full_frame_detector.cxx
  windowed_detector.cxx
  )


viame_install_headers(
  SUBDIR     viame/object_detectors
  ${plugin_headers}
  )

viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_object_detectors_export.h
  NOPATH   SUBDIR     viame/object_detectors
  )

viame_add_library( viame_object_detectors
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
                       viame_image_ops
                       # codecs/, for the debug frames the motion detector
                       # saves when a debug_dir is configured
                       viame_video_io
  )


set_target_properties( viame_object_detectors PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

viame_register_statically( viame_object_detectors
  register_algorithms.cxx
  )


###
# darknet's detector
#
# `lite-library-layout.md` puts it in this directory, and it was
# `plugins/darknet` until P7-T08's follow-up moved it here. Its trainer was
# removed when darknet was vendored for inference only.
#
# **Its own library rather than part of `viame_object_detectors`**, which is
# not tidiness. darknet is a large optional dependency, and
# `viame_object_detectors` is what `plugins/core` links for `windowed_utils`
# -- so folding darknet into it put libdarknet on the load path of
# everything that links `viame_core`, and back when darknet still carried
# OpenCV a gtest touching none of it failed to start for want of
# `libopencv_highgui`. An optional dependency belongs behind its own door.
##
if( VIAME_ENABLE_DARKNET )
  viame_install_headers(
    SUBDIR     viame/object_detectors
    darknet_detector.h
    )

  viame_install_headers(
    ${CMAKE_CURRENT_BINARY_DIR}/viame_object_detectors_darknet_export.h
    NOPATH   SUBDIR     viame/object_detectors
    )

  viame_add_library( viame_object_detectors_darknet
    darknet_detector.h
    darknet_detector.cxx
    )

  target_include_directories( viame_object_detectors_darknet PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    )

  target_link_libraries( viame_object_detectors_darknet
    PUBLIC               kwiver::vital
                         kwiver::vital_algo
                         kwiver::vital_config
                         kwiver::vital_logger
                         viame_object_detectors
    PRIVATE              $<BUILD_INTERFACE:viame::darknet>
                         viame_image_ops
    )

  set_target_properties( viame_object_detectors_darknet PROPERTIES
    SOVERSION            ${VIAME_VERSION_MAJOR}
    )

  viame_register_statically( viame_object_detectors_darknet
    register_darknet.cxx
    )

endif()

###
# Processes
##
set( process_headers
  detect_in_subregions_process.h
  detect_motion_process.h
  image_object_detector_process.h
  )

set( process_sources
  register_processes.cxx
  detect_in_subregions_process.cxx
  detect_motion_process.cxx
  image_object_detector_process.cxx
  )

viame_add_plugin( viame_processes_object_detectors
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
                   # `detect_in_subregions` chips the regions it is given
                   viame_image_ops
                   viame_object_detectors
  )

###
# Python implementations
##
if( VIAME_ENABLE_PYTHON )
  # Everything in the directory: `hough_circle_detector`, which was C++
  # until P7-T04 -- the Hough gradient transform is not an image_ops
  # primitive, so it stays cv2's -- and the GMM motion detector
  # (`stereo_algos`, `stereo_processes`) from `plugins/opencv` in P2-T05.
  viame_add_python_package( object_detectors )
endif()

target_include_directories( viame_processes_object_detectors PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
