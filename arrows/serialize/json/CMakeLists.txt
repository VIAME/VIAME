# Build / Install plugin for serialization

set( headers_public

  bounding_box.h
  detected_object.h
  detected_object_set.h
  detected_object_type.h
  )

set( sources

  bounding_box.cxx
  detected_object.cxx
  detected_object_set.cxx
  detected_object_type.cxx
  )

kwiver_add_library( kwiver_algo_serialize_json
  ${headers_public}
  ${sources}
  )

target_link_libraries( kwiver_algo_serialize_json
  PUBLIC               vital
  PRIVATE              vital_algo
  )

kwiver_install_headers(
  SUBDIR     arrows/serialize/json
  ${headers_public}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_serialize_json_export.h
  NOPATH   SUBDIR     arrows/serialize/json
  )

algorithms_create_plugin( kwiver_algo_serialize_json
  register_algorithms.cxx
  )

if (KWIVER_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
