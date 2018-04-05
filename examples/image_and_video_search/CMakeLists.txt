###
# Example image and appearance querying examples
##

if( VIAME_ENABLE_SMQTK )

  add_subdirectory( smqtk_on_chips )

  if( VIAME_ENABLE_VIVIA )
    add_subdirectory( video_ingest )
  endif()

endif()
