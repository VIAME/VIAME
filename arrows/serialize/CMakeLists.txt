# Build / Install plugin for serialization


if( KWIVER_ENABLE_SERIALIZE_PROTOBUF )
  add_subdirectory( protobuf )
endif()

if( KWIVER_ENABLE_SERIALIZE_JSON )
  add_subdirectory( json )
endif()
