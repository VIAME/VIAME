# Build / Install plugin for serialization

find_package(Protobuf MODULE REQUIRED)

if(NOT PROTOBUF_FOUND)
  message(FATAL_ERROR "-- Protobuf is required!")
endif()

set( headers_public
  bounding_box.h
  detected_object_type.h
  )

set( sources
  bounding_box.cxx
  detected_object_type.cxx
  )

link_directories(${PROTOBUF_LIBRARY})
include_directories(${PROTOBUF_INCLUDE_DIR})

kwiver_add_library( kwiver_serialize_proto
  ${headers_public}
  ${sources}
  )

target_link_libraries( kwiver_serialize_proto
  PUBLIC               vital
                       vital_algo
                       vital_types_protobuf
  )

kwiver_install_headers(
  SUBDIR     arrows/serialization/protobuf
  ${headers_public}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_serialize_proto_export.h
  NOPATH   SUBDIR     arrows/serialization/protobuf
  )

algorithms_create_plugin( kwiver_serialize_proto
  register_algorithms.cxx
  )

if (KWIVER_ENABLE_TESTS)
  ## add_subdirectory(tests)
endif()
