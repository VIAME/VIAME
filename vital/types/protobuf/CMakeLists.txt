#
# CMake file for building protobuf libraries.
#

find_package( Protobuf REQUIRED )

set( proto_files
  bounding_box.proto
  detected_object_type.proto
  detected_object.proto
  )

include_directories(${PROTOBUF_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_compile_options( -include "${CMAKE_CURRENT_BINARY_DIR}/vital_types_protobuf_export.h" )

PROTOBUF_GENERATE_CPP( cpp_srcs cpp_hdrs EXPORT_MACRO VITAL_TYPES_PROTOBUF_EXPORT ${proto_files} )

kwiver_add_library( vital_types_protobuf
  ${cpp_hdrs} ${cpp_srcs} )

target_link_libraries( vital_types_protobuf
  PUBLIC       vital
               ${PROTOBUF_LIBRARIES}
  )
