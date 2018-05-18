
###
# Download and install enabled model files
##
if( VIAME_DOWNLOAD_MODELS )

  # Video Search Models
  DownloadAndExtract(
    https://data.kitware.com/api/v1/item/59ae44048d777f7d33e9cf4a/download
    8d4eacbf237fbeb4e47b9f8c6eca896c
    ${VIAME_DOWNLOAD_DIR}/video_search_models_v0.1.0.tar.gz
    ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )

  # Default Caffe Models
  if( VIAME_ENABLE_CAFFE )
    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5a83a7d88d777f0685782a56/download
      2cdbe7ca51b49b12e59911aadec20e98
      ${VIAME_DOWNLOAD_DIR}/default_caffe_models_v0.1.0.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
  endif()

  # Default YOLO Configs
  if( VIAME_ENABLE_YOLO )
    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5a5e357d8d777f5e872f85c5/download
      8c4f120fb42b0ee579c770eec08304ea
      ${VIAME_DOWNLOAD_DIR}/yolo_models_v0.1.1.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )

    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5a5839ec8d777f5e872f8396/download
      64ed76909df326e6239742cbca78d7b7
      ${VIAME_DOWNLOAD_DIR}/yolo_v2_seed_model.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )

    if( VIAME_DOWNLOAD_MODELS-HABCAM )
      DownloadAndExtract(
        https://data.kitware.com/api/v1/item/5a5e39ad8d777f5e872f85c8/download
        41ef007ac005c1f202384b4478531238
        ${VIAME_DOWNLOAD_DIR}/habcam_models_v0.2.0.tar.gz
        ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
    endif()
  endif()
endif()

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
         DESTINATION configs 
