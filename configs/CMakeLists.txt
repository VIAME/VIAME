###
# All subdirectories for this folder
##
install( DIRECTORY gui-params
         DESTINATION configs )
install( DIRECTORY pipelines
         DESTINATION configs )
install( DIRECTORY prj-linux
         DESTINATION configs )
install( DIRECTORY prj-windows
         DESTINATION configs )

set( RUN_SCRIPT_FILES
     database_tool.py
     ingest_video.py
     launch_query_gui.py
     launch_track_viewer.py
     launch_annotation_gui.py )

install( FILES       ${RUN_SCRIPT_FILES}
         DESTINATION configs )

###
# Download and install enabled model files
##
if( VIAME_DOWNLOAD_MODELS )

endif()
