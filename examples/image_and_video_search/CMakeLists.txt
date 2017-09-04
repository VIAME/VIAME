###
# Example image and appearance querying examples
##

if( VIAME_ENABLE_SMQTK AND NOT WIN32 )

  add_subdirectory( smqtk_on_chips )

  if( VIAME_ENABLE_VIVIA )
    add_subdirectory( video_ingest )
  endif()

endif()
