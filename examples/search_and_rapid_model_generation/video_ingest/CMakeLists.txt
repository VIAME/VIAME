###
# Application level pipelines
##

set( EXAMPLE_INSTALL_FILES
     input_list.txt
     create_searchable_index.sh
     process_using_models.sh
     launch_query_gui.sh
     view_detections.prj )

###
# Install configuration files to examples directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/search_and_rapid_model_generation/video_ingest )
