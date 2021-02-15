# Build / Install plugin for serialization

set(CMAKE_FOLDER "Arrows/Serialize")


if( KWIVER_ENABLE_SERIALIZE_PROTOBUF )
  add_subdirectory( protobuf )
endif()

if( KWIVER_ENABLE_SERIALIZE_JSON )
  add_subdirectory( json )
endif()
