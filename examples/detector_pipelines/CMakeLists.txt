###
# Application level pipelines
##

set( EXAMPLE_INSTALL_FILES
     input_list.txt )

if( VIAME_ENABLE_YOLO )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       yolo_v2_detector.pipe
       habcam_detector.pipe )

  if( VIAME_DOWNLOAD_MODELS )
    DownloadAndExtract(
      https://data.kitware.com/api/v1/file/58c4b8848d777f0aef5d796b/download
      c10bf3f5be72d3d22a417f5f475604ad
      ${VIAME_DOWNLOAD_DIR}/yolo_models_v0.1.0.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR} )

    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/595a6bd18d777f16d01e9936/download
      e3a3931b8a802b26c39f2873e91002ba
      ${VIAME_DOWNLOAD_DIR}/habcam_models_v0.1.0.tar.gz
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

if( VIAME_ENABLE_SCALLOP_TK AND VIAME_ENABLE_YOLO AND VIAME_ENABLE_FASTER_RCNN )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       multi_cnn_detector.pipe )
endif()

if( VIAME_ENABLE_MATLAB AND VIAME_ENABLE_UW_PREDICTOR )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       camtrawl_pipeline.pipe )
endif()

###
# Install pipelines to example directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/detector_pipelines )
install( DIRECTORY   models
         DESTINATION examples/detector_pipelines )
install( FILES       output/example_detections.kw18
                     output/example_tot_scores.txt
                     output/yolo_v2_detections.prj
         DESTINATION examples/detector_pipelines/output )
