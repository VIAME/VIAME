###
# Example image and appearance querying examples
##

set( EXAMPLE_INSTALL_FILES
  input_list.txt )

if( WIN32 )
  set( EXAMPLE_INSTALL_FILES
    ${EXAMPLE_INSTALL_FILES}
    ingest_image_folder.bat
    ingest_using_detections.bat
    ingest_using_windowing.bat
    launch_gui_backend.bat )
else()
  set( EXAMPLE_INSTALL_FILES
    ${EXAMPLE_INSTALL_FILES}
    ingest_image_folder.sh
    ingest_using_detections.sh
    ingest_using_windowing.sh
    launch_gui_backend.sh
    db_tool.sh
    sql_init_table.sql
    reset_database.sh )
endif()

install( FILES      ${EXAMPLE_INSTALL_FILES}
   DESTINATION      examples/search_and_rapid_model_generation/smqtk_on_chips )
install( DIRECTORY  configs
   DESTINATION      examples/search_and_rapid_model_generation/smqtk_on_chips )
