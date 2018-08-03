###
# Application level pipelines
##

set( EXAMPLE_INSTALL_FILES
     README.rst
     input_list.txt
     view_detections.prj
     view_tracks.prj )

if( WIN32 )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       create_index.from_detections.bat
       create_index.full_frame_only.bat
       create_index.image_list.bat
       create_index.video_folder.bat
       process_database_using_models.bat
       process_ff_using_models.bat
       process_list_using_models.bat
       launch_query_gui.bat
       launch_annotation_gui.bat )
else()
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       create_index.from_detections.sh
       create_index.full_frame_only.sh
       create_index.image_list.sh
       create_index.video_folder.sh
       process_database_using_models.sh
       process_ff_using_models.sh
       process_list_using_models.sh
       launch_query_gui.sh
       launch_annotation_gui.sh )
endif()

###
# Install configuration files to examples directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/search_and_rapid_model_generation/viame_ingest )
