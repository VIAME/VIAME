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
       run_trained_model.bat
       train_default.bat )
else()
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       run_trained_model.sh
       train_default.sh )
endif()

if( VIAME_ENABLE_DARKNET )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_darknet_yolo.bat
         train_darknet_yolo_habcam_csv.bat
         train_darknet_yolo_kw18.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_darknet_yolo.sh
         train_darknet_yolo_habcam_csv.sh
         train_darknet_yolo_kw18.sh )
  endif()
endif()

if( VIAME_ENABLE_PYTORCH-NETHARN )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         continue_training_cfrnn.bat
         train_netharn_cfrnn.bat
         train_netharn_cfrnn_habcam_csv.bat
         train_netharn_mask_rcnn.bat
         train_netharn_motion_cfrnn.bat
         train_netharn_rf_detr.bat
         train_netharn_mit_yolo.bat
         train_netharn_rf_detr_16bit.bat
         input_list_arctic_seal_16bit.txt
         groundtruth_arctic_seal_16bit.csv
         labels_arctic_seal_16bit.txt )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         continue_training_cfrnn.sh
         train_netharn_cfrnn.sh
         train_netharn_cfrnn_habcam_csv.sh
         train_netharn_mask_rcnn.sh
         train_netharn_motion_cfrnn.sh
         train_netharn_rf_detr.sh
         train_netharn_mit_yolo.sh
         train_netharn_rf_detr_16bit.sh
         input_list_arctic_seal_16bit.txt
         groundtruth_arctic_seal_16bit.csv
         labels_arctic_seal_16bit.txt )
  endif()
endif()

if( VIAME_ENABLE_PYTORCH-RF-DETR )
  if( NOT WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_rf_detr.sh )
  endif()
endif()

if( VIAME_ENABLE_PYTORCH-MIT-YOLO )
  if( NOT WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_mit_yolo.sh )
  endif()
endif()

if( VIAME_ENABLE_PYTORCH-DETECTRON2 )
  if( NOT WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_detectron2_frcnn.sh )
  endif()
endif()

if( VIAME_ENABLE_PYTORCH-LITDET )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_litdet_frcnn.bat
         train_litdet_ssd.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_litdet_frcnn.sh
         train_litdet_ssd.sh )
  endif()
endif()

if( VIAME_ENABLE_SVM )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_svm_over_fish_dets.bat
         train_svm_over_generic_dets.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_svm_over_fish_dets.sh
         train_svm_over_generic_dets.sh )
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
