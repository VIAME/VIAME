###
# Detector training utilities
##

# Download formatted example training data
DownloadAndExtract(
  https://data.kitware.com/api/v1/item/5a5839318d777f5e872f8393/download
  ea0553ee4cc14f5941ad055a4468146c
  ${VIAME_DOWNLOAD_DIR}/training_example_set1.tar.gz
  ${CMAKE_CURRENT_SOURCE_DIR} )

# Install remaining example files
set( EXAMPLE_INSTALL_FILES
     README.rst )

if( VIAME_ENABLE_YOLO )
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
install( DIRECTORY   training_data
         DESTINATION examples/object_detector_training )
install( DIRECTORY   old_api
         DESTINATION examples/object_detector_training )
