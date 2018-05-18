###
# Detector training utilities
##

set( EXAMPLE_INSTALL_FILES
     README.rst )

DownloadAndExtract(
  https://data.kitware.com/api/v1/item/5a5839318d777f5e872f8393/download
  ea0553ee4cc14f5941ad055a4468146c
  ${VIAME_DOWNLOAD_DIR}/training_example_set1.tar.gz
  ${CMAKE_CURRENT_SOURCE_DIR} )

if( VIAME_ENABLE_YOLO )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       yolo_v2_544_kw18.conf
       yolo_v2_high_res_kw18.conf
       yolo_v2_high_res_habcam.conf )

  if( VIAME_DOWNLOAD_MODELS )
    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5a5839ec8d777f5e872f8396/download
      64ed76909df326e6239742cbca78d7b7
      ${VIAME_DOWNLOAD_DIR}/yolo_v2_seed_model.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR} )
  endif()

  if( NOT WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_yolo_v2_habcam.sh
         train_yolo_v2_kw18.sh )
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
         DESTINATION examples/object_detector_training )
install( DIRECTORY   cfg
         DESTINATION examples/object_detector_training )
install( DIRECTORY   training_data
         DESTINATION examples/object_detector_training )
