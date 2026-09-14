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
# The empty and full-frame detectors came from the `core` plugin in P2-T05,
# and `windowed_detector` is the merge of the two chippers that were in
# the `core` plugin and the `opencv` plugin: one implementation, registered as
# `windowed` with `ocv_windowed` an alias. `windowed_utils` went down to
# `image_ops`, where the refiner and the trainer can reach it too. The
# pytorch detectors and the shared `base` helpers came from the `pytorch` plugin
# in P2-T05, the detectors as one python subpackage per option.

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
# the `darknet` plugin until P7-T08's follow-up moved it here. Its trainer was
# removed when darknet was vendored for inference only.
#
# **Its own library rather than part of `viame_object_detectors`**, which is
# not tidiness. darknet is a large optional dependency, and
# `viame_object_detectors` is what the `core` plugin links for `windowed_utils`
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
# Vertex AI
#
# Processes that send detection requests to, and submit training jobs for, a
# Google Vertex AI endpoint, from `vertex-ai` in P2-T08. The REST client they
# share is `utilities`' `viame_vertex_ai`, built only when OpenSSL is found.
##
if( VIAME_ENABLE_VERTEX_AI AND TARGET viame_vertex_ai )
  viame_add_plugin( viame_processes_vertex_ai
    KIND        PROCESSES
    SOURCES          register_processes_vertex_ai.cxx
                     vertex_ai_detector.cxx
                     vertex_ai_detector.h
                     vertex_ai_trainer.cxx
                     vertex_ai_trainer.h
    PRIVATE          viame_vertex_ai
                     kwiver::sprokit_pipeline
                     kwiver::vital
                     kwiver::vital_util
    )

  target_include_directories( viame_processes_vertex_ai PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    )
endif()

###
# Python implementations
##
if( VIAME_ENABLE_PYTHON )
  # Everything in the directory: `hough_circle_detector`, which was C++
  # until P7-T04 -- the Hough gradient transform is not an image_ops
  # primitive, so it stays cv2's -- and the GMM motion detector
  # (`stereo_algos`, `stereo_processes`) from the `opencv` plugin in P2-T05.
  viame_add_python_package( object_detectors
    EXCLUDE "^(onnx|mmdet|netharn|huggingface|mit_yolo|ultralytics|learn|rf_detr|litdet|detectron2)/" )

  # The onnxruntime detector, from the `onnx` plugin in P2-T05. Its own
  # subpackage so that a build without ONNX neither installs it nor
  # declares it -- which is what the `onnx` plugin not being added achieved.
  viame_add_python_package( object_detectors/onnx
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/onnx
    CONDITION VIAME_ENABLE_ONNX )

  # The pytorch detectors, from the `pytorch` plugin in P2-T05: one subpackage
  # per option, each declaring only what it installs.
  foreach( gate IN ITEMS
      "mmdet:MMDET" "netharn:NETHARN" "huggingface:HUGGINGFACE"
      "mit_yolo:MIT-YOLO" "ultralytics:ULTRALYTICS" "rf_detr:RF-DETR"
      "litdet:LITDET" "detectron2:DETECTRON2" )
    string( REPLACE ":" ";" gate "${gate}" )
    list( GET gate 0 subdir )
    list( GET gate 1 option )
    viame_add_python_package( object_detectors/${subdir}
      DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}
      CONDITION VIAME_ENABLE_PYTORCH VIAME_ENABLE_PYTORCH-${option} )
  endforeach()

  # Each backend's trainer, launcher and onnx exporter sit beside its
  # detector since P2-T07, under the same option; `netharn/netharn` and the
  # `learn` packages are the vendored trees the detectors and trainers share.
  #
  # `learn/` also carries the vendored ReMax tree the remax detectors build
  # on, which lite-library-layout.md sends to a wheel in P9. Its CUDA
  # extension's `setup.py` and `test.py` were never installed and are not now.
  viame_add_python_package( object_detectors/learn
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/learn
    EXCLUDE "^remax/model/ops/(setup|test)\\.py$"
            # of the vendored learn packages, only these were ever installed
            "^(pydensecrf|panopticapi)/"
    CONDITION VIAME_ENABLE_PYTORCH VIAME_ENABLE_PYTORCH-LEARN )
endif()

target_include_directories( viame_processes_object_detectors PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
