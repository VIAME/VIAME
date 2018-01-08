###
# Detector training utilities
##

set( EXAMPLE_INSTALL_FILES
     README.md )

if( VIAME_ENABLE_YOLO )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       dnet_544_kw18.conf
       dnet_high_res_kw18.conf
       dnet_high_res_habcam.conf )

  if( VIAME_DOWNLOAD_MODELS )
#    DownloadAndExtract(
#      https://data.kitware.com/api/v1/item/595a6ccf8d777f16d01e993a/download
#      e3a3931b8a802b26c39f2873e91002ba
#      ${VIAME_DOWNLOAD_DIR}/habcam_models_v0.1.0.tar.gz
#      ${CMAKE_CURRENT_SOURCE_DIR} )
  endif()

  if( NOT WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_dnet_habcam.sh
         train_dnet_kw18.sh )
  endif()
endif()

#if( VIAME_ENABLE_SCALLOP_TK )
#  set( EXAMPLE_INSTALL_FILES
#       ${EXAMPLE_INSTALL_FILES}
#       scallop_tk_detector.pipe )
#endif()

###
# Install pipelines to example directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/detector_training )
install( DIRECTORY   cfg
         DESTINATION examples/detector_training )
install( DIRECTORY   training_data
         DESTINATION examples/detector_training )
