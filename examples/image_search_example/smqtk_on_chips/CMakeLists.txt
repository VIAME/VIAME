###
# Example image and appearance querying examples
##

if( VIAME_ENABLE_SMQTK AND NOT WIN32 )

  set( example_files
    ingest_chip_folder.sh
    ingest_using_detections.sh
    ingest_using_windowing.sh
    launch_gui_backend.sh
    reset_database.sh )

  install( FILES      ${example_files}
     DESTINATION      examples/image_search_example/smqtk_on_chips )
  install( DIRECTORY  configs
     DESTINATION      examples/image_search_example/smqtk_on_chips )

endif()
