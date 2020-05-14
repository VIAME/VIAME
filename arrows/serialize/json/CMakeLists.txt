# Build / Install plugin for serialization

find_package(ZLIB MODULE REQUIRED)
include_directories( ${ZLIB_INCLUDE_DIRS} )

set( headers_public
  bounding_box.h
  detected_object.h
  detected_object_set.h
  detected_object_type.h
  image.h
  metadata.h
  object_track_set.h
  object_track_state.h
  string.h
  timestamp.h
  track.h
  track_set.h
  track_state.h
  )

set( private_headers
  load_save.h
  load_save_point.h
  load_save_track_state.h
  load_save_track_set.h
  track_item.h
  )

set( sources
  bounding_box.cxx
  detected_object.cxx
  detected_object_set.cxx
  detected_object_type.cxx
  image.cxx
  load_save.cxx
  load_save_point.cxx
  load_save_metadata.cxx
  metadata.cxx
  object_track_set.cxx
  object_track_state.cxx
  string.cxx
  timestamp.cxx
  track.cxx
  track_set.cxx
  track_state.cxx
  )

kwiver_install_headers(
  SUBDIR     arrows/serialize/json
  ${headers_public}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_serialize_json_export.h
  NOPATH   SUBDIR     arrows/serialize/json
  )

kwiver_add_library( kwiver_serialize_json
  ${headers_public}
  ${sources}
  )

target_link_libraries( kwiver_serialize_json
  PUBLIC               vital_algo
  PRIVATE              ${ZLIB_LIBRARIES}
  )


algorithms_create_plugin( kwiver_serialize_json
  register_algorithms.cxx
  )

if (KWIVER_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
