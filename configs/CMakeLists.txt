
###
# Download and install enabled model files
##
if( VIAME_DOWNLOAD_MODELS )

  # GUI Context View
  DownloadAndExtract(
    https://data.kitware.com/api/v1/item/5afe6c018d777f15ebe1d6f2/download
    7e8800cb11a1a491140e7f58e9053030
    ${VIAME_DOWNLOAD_DIR}/models-blue_marble_context-v1.0.0.tar.gz
    ${CMAKE_CURRENT_SOURCE_DIR}/gui-params )

  # PyTorch Tracking Models
  if( VIAME_ENABLE_PYTORCH AND VIAME_DOWNLOAD_MODELS-PYTORCH )
    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5b6890a38d777f06857c1f22/download
      51b979d61df667b4751000192426f106
      ${VIAME_DOWNLOAD_DIR}/models-pytorch_tracking-v1.1.0.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )

    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5afe6d678d777f15ebe1d6f6/download
      fadc40de1f50b7b502d4a9b1d809cf28
      ${VIAME_DOWNLOAD_DIR}/models-pytorch_descriptors-v1.0.0.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
  endif()

  # Default Caffe Models
  if( VIAME_ENABLE_CAFFE )
    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5afe6e1e8d777f15ebe1d6fc/download
      2cdbe7ca51b49b12e59911aadec20e98
      ${VIAME_DOWNLOAD_DIR}/models-default_caffe_descriptor-v1.0.0.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
  endif()

  # Default YOLO Configs
  if( VIAME_ENABLE_YOLO )
    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5b5f00d18d777f06857c17fa/download
      fad64978df6322605530a5cc6274467c
      ${VIAME_DOWNLOAD_DIR}/models-default_yolo_v2-v2.1.0.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )

    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5afe74e68d777f15ebe1d703/download
      989ee1e694b736edf08eb7ed2c0337ab
      ${VIAME_DOWNLOAD_DIR}/models-yolo_v2_seed_model-v1.0.0.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )

    if( VIAME_DOWNLOAD_MODELS-HABCAM )
      DownloadAndExtract(
        https://data.kitware.com/api/v1/item/5b5e27cc8d777f06857c17c2/download
        0ee03ac4a199cc6f884b2f9702e59eb1
        ${VIAME_DOWNLOAD_DIR}/models-habcam-v2.0.0.tar.gz
        ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
    endif()

    if( VIAME_DOWNLOAD_MODELS-MOUSS )
      DownloadAndExtract(
        https://data.kitware.com/api/v1/item/5b7216e68d777f06857c4557/download
        a9a65372d077da45ad037a5aac288b8b
        ${VIAME_DOWNLOAD_DIR}/models-mouss-v2.0.0.tar.gz
        ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
    endif()
  endif()

  # Default Faster-RCNN Models
  if( VIAME_ENABLE_FASTER_RCNN )
    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5afe74fc8d777f15ebe1d70c/download
      e4792d72721465a206580522c37d0000
      ${VIAME_DOWNLOAD_DIR}/models-default_faster_rcnn-v1.0.0.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
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
     launch_annotation_gui.py
     aggregate_plots.py )

install( FILES       ${RUN_SCRIPT_FILES}
         DESTINATION configs )
