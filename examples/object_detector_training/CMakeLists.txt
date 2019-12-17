###
# Detector training utilities
##

# Download formatted example training data
DownloadAndExtract(
  https://data.kitware.com/api/v1/item/5de6bc61af2e2eed3552525f/download
  fd6fc82188564b48656a61bf2378d456
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

if( VIAME_ENABLE_PYTORCH )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_cfrnn_viame_csv.bat
         train_cfrnn_habcam.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_cfrnn_viame_csv.sh
         train_cfrnn_habcam.sh )
  endif()

  if( VIAME_ENABLE_SMQTK )
    if( WIN32 )
      set( EXAMPLE_INSTALL_FILES
           ${EXAMPLE_INSTALL_FILES}
           train_svm_viame_csv.bat )
    else()
      set( EXAMPLE_INSTALL_FILES
           ${EXAMPLE_INSTALL_FILES}
           train_svm_viame_csv.sh )
    endif()
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