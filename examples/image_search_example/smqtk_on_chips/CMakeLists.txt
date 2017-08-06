###
# Example image and appearance querying examples
##

set( example_files
  chip_pipeline.pipe
  ingest_data.sh
  initialize_database.sh
  launch_iqr_session.sh
  reset_database.sh )

install( FILES      ${example_files}
   DESTINATION      examples/image_search_example/smqtk_on_chips )
install( DIRECTORY  configs
   DESTINATION      examples/image_search_example/smqtk_on_chips )
