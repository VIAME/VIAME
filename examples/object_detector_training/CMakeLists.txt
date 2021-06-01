###
# Detector training utilities
##

# Download formatted example training data
DownloadAndExtract(
  https://data.kitware.com/api/v1/item/60b5489d2fa25629b9f84808/download
  bce0f0ab345e1d10b720b1c3cb8799df
  ${VIAME_DOWNLOAD_DIR}/training_example_set2.tar.gz
  ${CMAKE_CURRENT_SOURCE_DIR} )

# Install remaining example files
set( EXAMPLE_INSTALL_FILES
     README.rst )

if( WIN32 )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       run_trained_model.bat )
else()
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       run_trained_model.sh )
endif()

if( VIAME_ENABLE_DARKNET )
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

if( VIAME_ENABLE_PYTORCH-NETHARN )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         continue_training_cfrnn.bat
         train_cfrnn_viame_csv.bat
         train_cfrnn_habcam.bat
         train_mask_rcnn_viame_csv.bat
         train_motion_cfrnn_viame_csv.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         continue_training_cfrnn.sh
         train_cfrnn_viame_csv.sh
         train_cfrnn_habcam.sh
         train_mask_rcnn_viame_csv.sh
         train_motion_cfrnn_viame_csv.sh )
  endif()
endif()

if( VIAME_ENABLE_SMQTK )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_fish_svm_viame_csv.bat
         train_generic_svm_viame_csv.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_fish_svm_viame_csv.sh
         train_generic_svm_viame_csv.sh )
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
