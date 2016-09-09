
###
# add all plugin subdirectories below

add_subdirectory( core )

if( VIAME_ENABLE_SCALLOP_TK )
  add_subdirectory( scallop_tk )
endif()
