###
# Application level pipelines
##

set( EXAMPLE_INSTALL_FILES
     input_list.txt
     build_index.sh
     db_tool.sh
     ingest_video.sh
     launch_gui.sh )

if( VIAME_DOWNLOAD_MODELS )
#  DownloadAndExtract(
#    https://data.kitware.com/api/v1/file/58c4b8848d777f0aef5d796b/download
#    c10bf3f5be72d3d22a417f5f475604ad
#    ${VIAME_DOWNLOAD_DIR}/yolo_models_v0.1.0.tar.gz
#    ${CMAKE_CURRENT_SOURCE_DIR} )
endif()

###
# Install configuration files to examples directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/image_and_video_search/video_ingest )
install( DIRECTORY   configs
         DESTINATION examples/image_and_video_search/video_ingest )
