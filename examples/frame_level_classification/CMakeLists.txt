###
# Classifier training example
##

# Download formatted example training data
DownloadAndExtract(
  https://data.kitware.com/api/v1/item/5e6008f3af2e2eed35fd0453/download
  6b5e12186fc49c2088b0069ff48e8ee4
  ${VIAME_DOWNLOAD_DIR}/training_example_set1.tar.gz
  ${CMAKE_CURRENT_SOURCE_DIR} )

# Install remaining example files
set( EXAMPLE_INSTALL_FILES
     README.rst )

if( VIAME_ENABLE_PYTORCH-NETHARN )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         continue_training_deep_resnet.bat
         run_trained_deep_model.bat
         train_deep_resnet_viame_csv.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         continue_training_deep_resnet.sh
         run_trained_deep_model.sh
         train_deep_resnet_viame_csv.sh )
  endif()
endif()

if( VIAME_ENABLE_SMQTK )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         run_trained_svm_model.bat
         train_svm_resnet_viame_csv.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         run_trained_svm_model.sh
         train_svm_resnet_viame_csv.sh )
  endif()
endif()

###
# Install pipelines to example directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/frame_level_classification )
install( DIRECTORY   training_data
         DESTINATION examples/frame_level_classification )
