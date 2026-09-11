# Reading and writing detections and tracks
#
# Imported from kwiver in P5-T04: the kw18 readers and writers from
# arrows/core, and the five detection and track processes from
# sprokit/processes/core. The code is unchanged apart from its include paths
# and its export macro, which is this library's now; the namespace is still
# kwiver's, and phase 8 is where that changes.
#
# VIAME's own file readers and writers -- viame_csv, coco, dive, cvat and the
# rest -- are still in plugins/core and join this in phase 2.
#
# `opencv_yaml` is not imported: it is P7-T05's replacement for
# `cv::FileStorage`, the reader and writer for the calibration file format.

set( plugin_headers
  detected_object_set_input_kw18.h
  detected_object_set_output_kw18.h
  feature_descriptor_io.h
  opencv_yaml.h
  read_object_track_set_kw18.h
  write_object_track_set_kw18.h
  )

set( plugin_sources
  detected_object_set_input_kw18.cxx
  detected_object_set_output_kw18.cxx
  feature_descriptor_io.cxx
  opencv_yaml.cxx
  read_object_track_set_kw18.cxx
  write_object_track_set_kw18.cxx
  )

kwiver_install_headers(
  SUBDIR     viame/file_io
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_file_io_export.h
  NOPATH   SUBDIR     viame/file_io
  )

kwiver_add_library( viame_file_io
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_file_io PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  # feature_descriptor_io serialises with cereal, which is vendored
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/cereal>
  )

target_link_libraries( viame_file_io
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
                       kwiver::kwiversys
  )

set_target_properties( viame_file_io PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

algorithms_create_plugin( viame_file_io
  register_algorithms.cxx
  )

target_link_libraries( viame_file_io_plugin
  PUBLIC               kwiver::vital_vpm
  )

###
# Processes
##
set( process_headers
  detected_object_input_process.h
  detected_object_output_process.h
  read_object_track_process.h
  write_object_track_process.h
  write_track_descriptor_process.h
  )

set( process_sources
  register_processes.cxx
  detected_object_input_process.cxx
  detected_object_output_process.cxx
  read_object_track_process.cxx
  write_object_track_process.cxx
  write_track_descriptor_process.cxx
  )

kwiver_add_plugin( viame_processes_file_io
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
                   viame_file_io
  )

target_include_directories( viame_processes_file_io PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

###
# Python bindings
#
# Only `opencv_yaml`, and only so the golden recording of what
# `cv::FileStorage` parsed can be replayed against the reader that replaced
# it. See `tests/golden/calib_runner.py`.
##
if( VIAME_ENABLE_PYTHON )
  kwiver_create_python_init( file_io )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    file_io
    __init__ )

  find_package( pybind11 REQUIRED )

  kwiver_add_python_library( _opencv_yaml
    file_io
    SOURCES opencv_yaml_python.cxx )

  # `${PYTHON_LIBRARIES}` as well as pybind11: VIAME links with
  # `-Wl,--no-undefined`, so an extension module that leaves the interpreter
  # symbols to be resolved at load time does not link at all here.
  target_link_libraries( python-file_io-_opencv_yaml
    PRIVATE ${PYTHON_LIBRARIES} pybind11::pybind11 viame_file_io )
endif()
