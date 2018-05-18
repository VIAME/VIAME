###
# Application level pipelines
##

set( EXAMPLE_INSTALL_FILES
     input_list.txt
     ingest_list.sh
     view_detections.prj )

###
# Install configuration files to examples directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/search_and_rapid_model_generation/video_ingest )
