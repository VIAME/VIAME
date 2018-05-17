###
# Example image and appearance querying examples
##

set( EXAMPLE_INSTALL_FILES
  input_list.txt
  ingest_image_folder.sh
  ingest_using_detections.sh
  ingest_using_windowing.sh
  launch_gui_backend.sh
  reset_database.sh )

install( FILES      ${EXAMPLE_INSTALL_FILES}
   DESTINATION      examples/image_and_video_search/smqtk_on_chips )
install( DIRECTORY  configs
   DESTINATION      examples/image_and_video_search/smqtk_on_chips )
