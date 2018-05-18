###
# Application level pipelines
##

set( EXAMPLE_INSTALL_FILES
     input_list.txt
     habcam_list.txt )

if( VIAME_ENABLE_YOLO )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       yolo_v2_detector.pipe
       habcam_detector.pipe )

  if( VIAME_DOWNLOAD_MODELS )
    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5a5e357d8d777f5e872f85c5/download
      8c4f120fb42b0ee579c770eec08304ea
      ${VIAME_DOWNLOAD_DIR}/yolo_models_v0.1.1.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR} )

    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5a5e39ad8d777f5e872f85c8/download
      41ef007ac005c1f202384b4478531238
      ${VIAME_DOWNLOAD_DIR}/habcam_models_v0.2.0.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR} )
  endif()

  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         run_yolo.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         run_yolo.sh )
  endif()

  if( NOT WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         run_habcam.sh )
  endif()
endif()

if( VIAME_ENABLE_SCALLOP_TK )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       scallop_tk_detector.pipe )
endif()

if( VIAME_ENABLE_OPENCV )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       hough_detector.pipe )
endif()

if( VIAME_ENABLE_SCALLOP_TK AND VIAME_ENABLE_YOLO )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       multi_cnn_detector.pipe )
endif()

if( VIAME_ENABLE_MATLAB )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       lanl_detector.pipe )

  if( VIAME_ENABLE_UW_PREDICTOR )
    set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       camtrawl_pipeline.pipe )
  endif()

  if( NOT WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         run_lanl.sh )
  endif()
endif()

###
# Install pipelines to example directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/object_detection )
install( DIRECTORY   models
         DESTINATION examples/object_detection )
install( FILES       output/example_detections.kw18
                     output/example_tot_scores.txt
                     output/yolo_v2_detections.prj
         DESTINATION examples/object_detection/output )
