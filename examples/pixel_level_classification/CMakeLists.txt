DownloadAndExtract(
  https://data.kitware.com/api/v1/item/5ff32fed2fa25629b9bdaf57/download
  c5c94ec26275e4c0d4804800b82323b8
  ${VIAME_DOWNLOAD_DIR}/pixel_classification_fish_set1.tar.gz
  ${CMAKE_CURRENT_SOURCE_DIR} )

# Install remaining example files
set( EXAMPLE_INSTALL_FILES
     input_images.txt
     README.rst )

if( WIN32 )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       train_adaboost_pixel_classifier.bat )
else()
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       train_adaboost_pixel_classifier.sh )
endif()

install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/pixel_level_classification )
install( DIRECTORY   training_data
         DESTINATION examples/pixel_level_classification )
