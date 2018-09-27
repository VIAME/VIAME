###
# Detector training utilities
##

# Download formatted example training data
DownloadAndExtract(
  https://data.kitware.com/api/v1/item/5afed6f68d777f15ebe1d795/download
  e38843185bcb4f56f85ff208943cddc3
  ${VIAME_DOWNLOAD_DIR}/training_example_set1.tar.gz
  ${CMAKE_CURRENT_SOURCE_DIR} )

# Install remaining example files
set( EXAMPLE_INSTALL_FILES
     README.rst )

if( VIAME_ENABLE_YOLO )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_yolo_habcam.bat
         train_yolo_viame_csv.bat
         train_yolo_kw18.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_yolo_habcam.sh
         train_yolo_viame_csv.sh
         train_yolo_kw18.sh )
  endif()
endif()

###
# Install pipelines to example directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/object_detector_training )
install( DIRECTORY   training_data_habcam
         DESTINATION examples/object_detector_training )
install( DIRECTORY   training_data_mouss
         DESTINATION examples/object_detector_training )
install( DIRECTORY   deprecated_old_api
         DESTINATION examples/object_detector_training )
