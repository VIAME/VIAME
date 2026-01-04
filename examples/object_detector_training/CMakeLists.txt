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
         train_yolo_from_habcam_csv.bat
         train_yolo_from_viame_csv.bat
         train_yolo_from_kw18.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_yolo_from_habcam_csv.sh
         train_yolo_from_viame_csv.sh
         train_yolo_from_kw18.sh )
  endif()
endif()

if( VIAME_ENABLE_PYTORCH-NETHARN )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         continue_training_cfrnn.bat
         train_cfrnn_from_viame_csv.bat
         train_cfrnn_from_habcam_csv.bat
         train_mask_rcnn_from_viame_csv.bat
         train_motion_cfrnn_from_viame_csv.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         continue_training_cfrnn.sh
         train_cfrnn_from_viame_csv.sh
         train_cfrnn_from_habcam_csv.sh
         train_mask_rcnn_from_viame_csv.sh
         train_motion_cfrnn_from_viame_csv.sh )
  endif()
endif()

if( VIAME_ENABLE_SVM )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_svm_over_fish_dets_from_viame_csv.bat
         train_svm_over_generic_dets_from_viame_csv.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_svm_over_fish_dets_from_viame_csv.sh
         train_svm_over_generic_dets_from_viame_csv.sh )
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
