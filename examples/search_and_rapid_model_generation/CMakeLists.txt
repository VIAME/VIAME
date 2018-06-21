###
# Example image and appearance querying examples
##

if( VIAME_ENABLE_SMQTK )

  add_subdirectory( smqtk_on_chips )

  if( VIAME_ENABLE_VIVIA )
    add_subdirectory( viame_ingest )
  endif()

endif()

if( VIAME_ENABLE_FLASK )

  add_subdirectory( flask )

endif()
