# Build / Install Plugin containing core algorithm implementations

set( plugin_headers
  adaptive_detector_trainer.h
  adaptive_tracker_trainer.h
  evaluate_models.h
  measurement_utilities.h
  pair_stereo_detections.h
  pair_stereo_tracks.h
  utilities_training.h
  windowed_trainer.h
)

set( plugin_sources
  adaptive_detector_trainer.cxx
  adaptive_tracker_trainer.cxx
  evaluate_models.cxx
  measurement_utilities.cxx
  pair_stereo_detections.cxx
  pair_stereo_tracks.cxx
  utilities_training.cxx
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
  calibrate_cameras_from_tracks_process.h
  filter_frame_index_process.h
  filter_frame_process.h
  image_to_image_set_process.h
  measure_objects_process.h
  pair_stereo_detections_process.h
  refine_measurements_process.h
)

set( process_sources
  register_processes.cxx
  calibrate_cameras_from_tracks_process.cxx
  filter_frame_index_process.cxx
  filter_frame_process.cxx
  image_to_image_set_process.cxx
  measure_objects_process.cxx
  pair_stereo_detections_process.cxx
  refine_measurements_process.cxx
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
  )

if( VIAME_ENABLE_PYTHON )

  viame_create_python_init( core )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    core
    __init__ )

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
    ${CMAKE_CURRENT_SOURCE_DIR}/frame_diff_trainer.py
    core
    frame_diff_trainer )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/ocsort_trainer.py
    core
    ocsort_trainer )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/survey_metadata.py
    core
    survey_metadata )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/curved_measurement.py
    core
    curved_measurement )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/interactive_stereo.py
    core
    interactive_stereo )

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

