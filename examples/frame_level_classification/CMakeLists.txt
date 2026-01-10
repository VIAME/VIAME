###
# Classifier training example
##

# Install remaining example files
set( EXAMPLE_INSTALL_FILES
     README.rst
     input_list.txt
     labels.txt )

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

###
# Custom install of file to example images folder
##
install( FILES       fish_no_fish.csv
         DESTINATION examples/example_imagery/small_example_image_set1 )
