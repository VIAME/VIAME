# Build / Install plugin for serialization

set( headers_public
  bounding_box.h
  )

set( sources
  bounding_box.cxx
  )


kwiver_add_library( kwiver_serialize_proto
  ${headers_public}
  ${sources}
  )

target_link_libraries( kwiver_serialize_proto
  PUBLIC               vital_algo
  )

kwiver_install_headers(
  SUBDIR     arrows/serialization
  ${headers_public}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_serialize_proto_export.h
  NOPATH   SUBDIR     arrows/serialization
  )

algorithms_create_plugin( kwiver_serialize_proto
  register_algorithms.cxx
  )

if (KWIVER_ENABLE_TESTS)
  ## add_subdirectory(tests)
endif()
