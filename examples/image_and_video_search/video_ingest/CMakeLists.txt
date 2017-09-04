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
  DownloadAndExtract(
    https://data.kitware.com/api/v1/item/59adcadc8d777f7d33e9cf2d/download
    4bbb8608f979fdbb00683b5ef689573d
    ${VIAME_DOWNLOAD_DIR}/video_search_models_v0.1.0.tar.gz
    ${CMAKE_CURRENT_SOURCE_DIR}/configs )
endif()

###
# Install configuration files to examples directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/image_and_video_search/video_ingest )
install( DIRECTORY   configs
         DESTINATION examples/image_and_video_search/video_ingest )
