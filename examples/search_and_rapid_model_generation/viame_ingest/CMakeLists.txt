###
# Application level pipelines
##

set( EXAMPLE_INSTALL_FILES
     README.rst
     input_list.txt
     create_index.image_list.sh
     create_index.video_folder.sh
     process_using_models.sh
     launch_query_gui.sh
     view_detections.prj )

###
# Install configuration files to examples directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/search_and_rapid_model_generation/viame_ingest )
