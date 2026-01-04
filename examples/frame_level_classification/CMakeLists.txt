###
# Classifier training example
##

# Download formatted example training data and setup
DownloadAndExtract(
  https://data.kitware.com/api/v1/file/58c4b9458d777f0aef5d796e/download
  8a5a40018e3909d10b3fd50aad06e4f3
  ${VIAME_DOWNLOAD_DIR}/small_example_image_set1.tar.gz
  ${CMAKE_CURRENT_SOURCE_DIR}/training_data )

file( COPY labels.txt
      DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/training_data )
file( COPY fish_no_fish.csv
      DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/training_data/small_example_image_set1 )

# Install remaining example files
set( EXAMPLE_INSTALL_FILES
     README.rst
     input_list.txt )

if( WIN32 )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       run_trained_model.bat )
else()
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       run_trained_model.sh )
endif()

if( VIAME_ENABLE_PYTORCH-NETHARN )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         continue_training_deep_classifier.bat
         train_deep_frame_classifier.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         continue_training_deep_classifier.sh
         train_deep_frame_classifier.sh )
  endif()
endif()

if( VIAME_ENABLE_SVM )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_svm_frame_classifier.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_svm_frame_classifier.sh )
  endif()
endif()

###
# Install pipelines to example directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/frame_level_classification )
install( DIRECTORY   training_data
         DESTINATION examples/frame_level_classification )
