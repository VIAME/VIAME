# Build / Install Plugin containing core algorithm implementations

set( plugin_headers
  adaptive_detector_trainer.h
  adaptive_tracker_trainer.h
  average_track_descriptors.h
  convert_head_tail_points.h
  empty_detector.h
  evaluate_models.h
  full_frame_detector.h
  measurement_utilities.h
  merge_detections_suppress_in_regions.h
  equalize_via_percentiles.h
  pair_stereo_detections.h
  pair_stereo_tracks.h
  query_track_descriptor_set_csv.h
  refine_detections_add_fixed.h
  refine_detections_nms.h
  refine_tracks_average_tot.h
  utilities_training.h
  windowed_detector.h
  windowed_refiner.h
  windowed_trainer.h
)

set( plugin_sources
  adaptive_detector_trainer.cxx
  adaptive_tracker_trainer.cxx
  average_track_descriptors.cxx
  convert_head_tail_points.cxx
  empty_detector.cxx
  evaluate_models.cxx
  full_frame_detector.cxx
  measurement_utilities.cxx
  merge_detections_suppress_in_regions.cxx
  equalize_via_percentiles.cxx
  pair_stereo_detections.cxx
  pair_stereo_tracks.cxx
  query_track_descriptor_set_csv.cxx
  refine_detections_add_fixed.cxx
  refine_detections_nms.cxx
  refine_tracks_average_tot.cxx
  utilities_training.cxx
  windowed_detector.cxx
  windowed_refiner.cxx
  windowed_trainer.cxx
)

viame_install_headers(
  SUBDIR     viame
  ${plugin_headers}
  )

viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_core_export.h
  NOPATH   SUBDIR     viame
  )

viame_add_library( viame_core
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

set( CORE_LINK_LIBRARIES
  viame_utilities
  kwiver::vital
  kwiver::vital_algo
  kwiver::vital_applets
  kwiver::vital_config
  kwiver::vital_exceptions
  kwiver::vital_logger
  kwiver::vital_util
  viame_measurement
  # `windowed_utils` moved to `library/object_detectors` when darknet did:
  # `library/` cannot reach up into `plugins/`, and darknet's detector needs
  # the chipping
  viame_object_detectors
  kwiver::sprokit_pipeline
  kwiver::kwiver_adapter
  )

# The polygon rasterisation the CSV parsers use is in-house now, and so is
# the OpenCV YAML `camera_rig_io` reads calibrations with (P7-T05)
# ... and the image writer the epipolar debug overlay uses (P7-T06)
set( AUX_LINK_LIBRARIES
     viame_image_ops viame_file_io viame_video_io ${AUX_LINK_LIBRARIES} )

# `plugins/core` has no OpenCV left as of P7-T07: the CSV writers' polygon
# conversion was the last of it, and `image_ops::find_borders` is what it is
# now. The `find_package` and the `VIAME_ENABLE_OPENCV` definition that stood
# here are gone with it.

# OpenMP is used for parallel evaluation in evaluate_models
find_package( OpenMP )
if( OpenMP_CXX_FOUND )
  set( AUX_LINK_LIBRARIES
       OpenMP::OpenMP_CXX
       ${AUX_LINK_LIBRARIES} )
endif()

# Python is used for DINOv3 feature matching via Python C API
if( VIAME_ENABLE_PYTHON )
  include_directories( SYSTEM ${PYTHON_INCLUDE_DIRS} )
  get_filename_component( PYTHON_LIBRARY_DIR "${PYTHON_LIBRARY}" DIRECTORY )
  set( AUX_LINK_LIBRARIES
       ${PYTHON_LIBRARIES}
       ${AUX_LINK_LIBRARIES} )
  set( AUX_COMPILE_DEFINITIONS VIAME_ENABLE_PYTHON ${AUX_COMPILE_DEFINITIONS} )
endif()

target_link_libraries( viame_core
  PUBLIC               ${CORE_LINK_LIBRARIES}
                       ${AUX_LINK_LIBRARIES}
  )

if( VIAME_ENABLE_PYTHON )
  target_link_directories( viame_core PUBLIC ${PYTHON_LIBRARY_DIR} )
endif()

# Propagate compile definitions to dependent targets (important for ABI compatibility)
if( AUX_COMPILE_DEFINITIONS )
  target_compile_definitions( viame_core PUBLIC ${AUX_COMPILE_DEFINITIONS} )
endif()

set_target_properties( viame_core PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

viame_register_statically( viame_core
  register_algorithms.cxx
  )


# Add process library
set( process_headers
  accumulate_image_statistics_process.h
  accumulate_object_tracks_process.h
  align_multimodal_imagery_process.h
  calibrate_cameras_from_tracks_process.h
  create_database_query_process.h
  extract_desc_ids_for_training_process.h
  fetch_descriptors_process.h
  filter_frame_index_process.h
  filter_frame_process.h
  filter_object_tracks_process.h
  image_to_image_set_process.h
  ingest_descriptors_process.h
  measure_objects_process.h
  object_track_descriptors_process.h
  pair_stereo_detections_process.h
  process_query_process_adaboost.h
  refine_measurements_process.h
  resample_object_tracks_process.h
  select_database_query_process.h
  split_tracks_to_feature_landmarks_process.h
  stack_frames_process.h
  track_conductor_process.h
  warp_detections_process.h
  warp_image_process.h
  write_query_results_as_tracks_process.h
)

set( process_sources
  register_processes.cxx
  accumulate_image_statistics_process.cxx
  accumulate_object_tracks_process.cxx
  align_multimodal_imagery_process.cxx
  calibrate_cameras_from_tracks_process.cxx
  create_database_query_process.cxx
  extract_desc_ids_for_training_process.cxx
  fetch_descriptors_process.cxx
  filter_frame_index_process.cxx
  filter_frame_process.cxx
  filter_object_tracks_process.cxx
  image_to_image_set_process.cxx
  ingest_descriptors_process.cxx
  measure_objects_process.cxx
  object_track_descriptors_process.cxx
  pair_stereo_detections_process.cxx
  process_query_process_adaboost.cxx
  refine_measurements_process.cxx
  resample_object_tracks_process.cxx
  select_database_query_process.cxx
  split_tracks_to_feature_landmarks_process.cxx
  stack_frames_process.cxx
  track_conductor_process.cxx
  warp_detections_process.cxx
  warp_image_process.cxx
  write_query_results_as_tracks_process.cxx
)

viame_add_plugin( viame_processes_core
  KIND        PROCESSES
  SOURCES          ${process_sources}
                   ${private_headers}
  PRIVATE          ${CORE_LINK_LIBRARIES}
                   kwiver::sprokit_pipeline
                   kwiver::kwiver_adapter
                   kwiver::vital_util
                   kwiver::vital_vpm
                   viame_measurement
                   viame_core
 )

target_include_directories( viame_processes_core PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  # `iqr_session.h` reaches its neighbours as `core/...`, which is how it
  # was written for the plugins that include it from outside
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/plugins>
  )

###
# The IQR AdaBoost session
#
# Its own library, and the only thing in VIAME that calls the interpreter
# from C++: it trains and scores through `viame.core.iqr_adaboost`, which is
# scikit-learn. Kept out of `viame_core` on purpose -- the library
# everything links stays free of libpython, and only what needs the model
# picks the dependency up.
#
# Without python it compiles to a stub that reports no model, and the base
# class falls back to similarity scoring against the positive exemplars, so
# the query still returns ranked results.
##
viame_install_headers(
  SUBDIR     viame
  iqr_session_adaboost.h
  )

viame_add_library( viame_iqr
  iqr_session_adaboost.h
  iqr_session_adaboost.cxx
  )

target_include_directories( viame_iqr PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/plugins>
  )

target_link_libraries( viame_iqr
  PUBLIC               kwiver::vital
                       kwiver::vital_logger
                       viame_core
  )

set_target_properties( viame_iqr PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

if( VIAME_ENABLE_PYTHON )
  # pybind11 is vendored in `library/tpl/pybind11`; the targets it used to
  # come from are defined there

  target_compile_definitions( viame_iqr PRIVATE VIAME_CORE_HAVE_PYTHON )

  # `${PYTHON_LIBRARIES}` as well as pybind11: VIAME links with
  # `-Wl,--no-undefined`, so leaving the interpreter symbols to be resolved
  # at load time does not link at all here.
  target_link_libraries( viame_iqr
    PRIVATE ${PYTHON_LIBRARIES} $<BUILD_INTERFACE:pybind11::pybind11> )
endif()

target_link_libraries( viame_processes_core PRIVATE viame_iqr )

if( VIAME_ENABLE_PYTHON )

  viame_create_python_init( core )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    core
    __init__ )


  # The model behind `process_query_adaboost`, called from
  # `iqr_session_adaboost.cxx`
  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/iqr_adaboost.py
    core
    iqr_adaboost )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/bytetrack_tracker.py
    core
    bytetrack_tracker )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/bytetrack_trainer.py
    core
    bytetrack_trainer )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/training_data.py
    core
    training_data )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/tracker_param_search.py
    core
    tracker_param_search )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/detection_fusion_core.py
    core
    detection_fusion_core )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/merge_detections_nms_fusion.py
    core
    merge_detections_nms_fusion )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/merge_detections_coverage_reinforce.py
    core
    merge_detections_coverage_reinforce )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/merge_detections_simple.py
    core
    merge_detections_simple )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/merge_tracks_tube_iou.py
    core
    merge_tracks_tube_iou )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/multicam_homog_blackout.py
    core
    multicam_homog_blackout )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/multicam_homog_det_suppressor.py
    core
    multicam_homog_det_suppressor )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/multicam_homog_mosaic.py
    core
    multicam_homog_mosaic )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/multicam_homog_tracker.py
    core
    multicam_homog_tracker )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/equalize_via_percentiles.py
    core
    equalize_via_percentiles )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/optical_flow.py
    core
    optical_flow )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/frame_diff_trainer.py
    core
    frame_diff_trainer )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/ocsort_tracker.py
    core
    ocsort_tracker )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/ocsort_trainer.py
    core
    ocsort_trainer )







  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/segmentation_utils.py
    core
    segmentation_utils )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/interactive_service.py
    core
    interactive_service )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/survey_metadata.py
    core
    survey_metadata )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/utility_processes.py
    core
    utility_processes )


  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/simple_homog_tracker.py
    core
    simple_homog_tracker )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/stabilize_many_images.py
    core
    stabilize_many_images )




  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/interactive_segmentation.py
    core
    interactive_segmentation )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/curved_measurement.py
    core
    curved_measurement )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/interactive_stereo.py
    core
    interactive_stereo )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/align_cameras_process.py
    core
    align_cameras_process )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/alignment_core.py
    core
    alignment_core )


  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/query_service.py
    core
    query_service )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/index_descriptors.py
    core
    index_descriptors )

  # Compiled bindings exposing the C++ stereo measurement (compute_stereo_measurement,
  # aggregate_lengths) to Python so the interactive stereo service reuses them rather
  # than duplicating the math. The bindings live at the bottom of measurement_utilities.cxx
  # guarded by VIAME_MEASUREMENT_PYTHON_BINDINGS; defining it here makes this build of the
  # file emit only the module (linking viame_core for the implementations).
  # pybind11 is vendored in `library/tpl/pybind11`; the targets it used to
  # come from are defined there
  viame_add_python_library( _measurement
    core
    SOURCES measurement_utilities.cxx )
  target_compile_definitions( python-core-_measurement
    PRIVATE VIAME_MEASUREMENT_PYTHON_BINDINGS )
  target_link_libraries( python-core-_measurement
    PRIVATE ${PYTHON_LIBRARIES} $<BUILD_INTERFACE:pybind11::pybind11> viame_core )

endif()

