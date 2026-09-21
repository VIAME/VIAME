# Reading and writing detections and tracks
#
# Imported from kwiver in P5-T04: the kw18 readers and writers from
# arrows/core, and the five detection and track processes from
# sprokit/processes/core. The code is unchanged apart from its include paths
# and its export macro, which is this library's now; the namespace is still
# kwiver's, and phase 8 is where that changes.
#
# VIAME's own file readers and writers -- viame_csv, coco, dive, cvat and the
# rest -- joined them from the `core` plugin in P2-T04, along with the camera
# and camera-rig readers, the homography reader and writer, and the
# annotation conversion the `viame convert` applet runs.
#
# `opencv_yaml` is not imported: it is P7-T05's replacement for
# `cv::FileStorage`, the reader and writer for the calibration file format.

set( plugin_headers
  auto_detect_transform.h
  camera_io.h
  camera_rig_io.h
  convert_annotations.h
  convert_notes_to_attributes.h
  detected_object_set_input_kw18.h
  detected_object_set_output_kw18.h
  feature_descriptor_io.h
  json.h
  opencv_yaml.h
  portable_binary.h
  read_detected_object_set_auto.h
  read_detected_object_set_cvat.h
  read_detected_object_set_dive.h
  read_detected_object_set_fishnet.h
  read_detected_object_set_habcam.h
  read_detected_object_set_oceaneyes.h
  read_detected_object_set_viame_csv.h
  read_detected_object_set_yolo.h
  read_object_track_set_auto.h
  read_object_track_set_dive.h
  read_object_track_set_kw18.h
  read_object_track_set_viame_csv.h
  read_transform_homography_json.h
  write_detected_object_set_dive.h
  write_detected_object_set_viame_csv.h
  write_object_track_set_dive.h
  write_object_track_set_kw18.h
  write_object_track_set_viame_csv.h
  )

set( plugin_sources
  auto_detect_transform.cxx
  camera_io.cxx
  camera_rig_io.cxx
  convert_annotations.cxx
  convert_notes_to_attributes.cxx
  detected_object_set_input_kw18.cxx
  detected_object_set_output_kw18.cxx
  feature_descriptor_io.cxx
  opencv_yaml.cxx
  read_detected_object_set_auto.cxx
  read_detected_object_set_cvat.cxx
  read_detected_object_set_dive.cxx
  read_detected_object_set_fishnet.cxx
  read_detected_object_set_habcam.cxx
  read_detected_object_set_oceaneyes.cxx
  read_detected_object_set_viame_csv.cxx
  read_detected_object_set_yolo.cxx
  read_object_track_set_auto.cxx
  read_object_track_set_dive.cxx
  read_object_track_set_kw18.cxx
  read_object_track_set_viame_csv.cxx
  read_transform_homography_json.cxx
  write_detected_object_set_dive.cxx
  write_detected_object_set_viame_csv.cxx
  write_object_track_set_dive.cxx
  write_object_track_set_kw18.cxx
  write_object_track_set_viame_csv.cxx
  )

viame_install_headers(
  SUBDIR     viame/file_io
  ${plugin_headers}
  )

viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_file_io_export.h
  NOPATH   SUBDIR     viame/file_io
  )

viame_add_library( viame_file_io
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_file_io PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  # `json.h` is the one door to rapidjson, which is vendored
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/tpl/rapidjson>
  )

target_link_libraries( viame_file_io
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
                       kwiver::vital_applets
                       viame_utilities
                       # The polygon-and-mask geometry the viame_csv and dive
                       # readers and writers run
                       viame_image_kernels
                       # `read_detected_object_set_habcam` takes the frame
                       # time out of the file name
                       viame_image_io
  )

# TinyXML is `read_detected_object_set_cvat.cxx`'s and nothing else's, and
# miniz is the NPZ `camera_rig_io` reads. Both are vendored and static, so
# `$<BUILD_INTERFACE:...>` as well as PRIVATE: a static library linked even
# privately is recorded in the target's interface, and `install( EXPORT )`
# then refuses it for not being in the export set.
target_link_libraries( viame_file_io
  PRIVATE              $<BUILD_INTERFACE:viame::tinyxml>
                       $<BUILD_INTERFACE:viame::miniz>
  )

# The guard around the NPZ code is vestigial -- zlib was found or not found,
# miniz always is -- and left alone rather than edited out of working code.
target_compile_definitions( viame_file_io PUBLIC VIAME_ENABLE_ZLIB )

set_target_properties( viame_file_io PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

viame_register_statically( viame_file_io
  register_algorithms.cxx
  )


###
# Processes
##
set( process_headers
  detected_object_input_process.h
  store_descriptors_csv.h
  write_homography_list_process.h
  detected_object_output_process.h
  read_object_track_process.h
  write_object_track_process.h
  write_track_descriptor_process.h
  )

set( process_sources
  register_processes.cxx
  detected_object_input_process.cxx
  store_descriptors_csv.cxx
  write_homography_list_process.cxx
  detected_object_output_process.cxx
  read_object_track_process.cxx
  write_object_track_process.cxx
  write_track_descriptor_process.cxx
  )

viame_add_plugin( viame_processes_file_io
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
                   kwiver::kwiver_adapter
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
  # The COCO readers and writers, `utilities_coco` and the package's own
  # declarations; `viame_add_python_package` globs the directory, and the
  # `_viame_package_init` inside it leaves this package's real `__init__.py`
  # alone. A bare `viame_create_python_init` beside it would defeat that
  # guard -- it writes its stub at configure time and the real file's copy
  # never runs. This directory got away with it; `image_io` did not.
  viame_add_python_package( file_io )

  # pybind11 is vendored in `library/tpl/pybind11`; the targets it used to
  # come from are defined there

  viame_add_python_library( _opencv_yaml
    file_io
    SOURCES opencv_yaml_python.cxx )

  # `${PYTHON_LIBRARIES}` as well as pybind11: VIAME links with
  # `-Wl,--no-undefined`, so an extension module that leaves the interpreter
  # symbols to be resolved at load time does not link at all here.
  viame_target_link_libraries( python-file_io-_opencv_yaml
    PRIVATE ${PYTHON_LIBRARIES} $<BUILD_INTERFACE:pybind11::pybind11> viame_file_io )
endif()

###
# Tests
##
if( VIAME_ENABLE_TESTS )
  add_subdirectory( tests )
endif()
