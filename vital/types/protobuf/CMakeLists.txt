#
# CMake file for building protobuf libraries.
#

find_package( Protobuf REQUIRED )

set( proto_files
  bounding_box.proto
  )

include_directories(${PROTOBUF_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

PROTOBUF_GENERATE_CPP( cpp_srcs cpp_hdrs ${proto_files} )

kwiver_add_library( vital_types_protobuf
  ${cpp_hdrs} ${cpp_srcs} )

target_link_libraries( vital_types_protobuf
  PUBLIC       vital
  PRIVATE      ${PROTOBUF_LIBRARIES}
  )
