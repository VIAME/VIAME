# Build / Install plugin for serialization

find_package(Protobuf MODULE REQUIRED)

if(NOT PROTOBUF_FOUND)
  message(FATAL_ERROR "-- Protobuf is required!")
endif()

set( headers_public
  bounding_box.h
  detected_object_type.h
  detected_object.h
  detected_object_set.h
  )

set( sources
  bounding_box.cxx
  detected_object_type.cxx
  detected_object.cxx
  detected_object_set.cxx
  )

link_directories(${PROTOBUF_LIBRARY})
include_directories(${PROTOBUF_INCLUDE_DIR})

# Need to add this include file to the compile because of a defficiency in the protobuf
# compiler. The compiler has the option to insert an export macro into class definitions
# but has no way of defining that macro. as of libprotoc 2.6.1 this is the accepted way
# of defining the export symbols.
add_compile_options( -include "${CMAKE_BINARY_DIR}/vital/types/protobuf/vital_types_protobuf_export.h" )

kwiver_add_library( kwiver_serialize_protobuf
  ${headers_public}
  ${sources}
  )

target_link_libraries( kwiver_serialize_protobuf
  PUBLIC               vital_algo
  PRIVATE              vital_exceptions
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

algorithms_create_plugin( kwiver_serialize_protobuf
  register_algorithms.cxx
  )

if (KWIVER_ENABLE_TESTS)
   add_subdirectory(tests)
endif()
