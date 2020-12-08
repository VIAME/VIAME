# TODO download the data from a server

# Install remaining example files
set( EXAMPLE_INSTALL_FILES
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
install( DIRECTORY   training_data_adaboost
         DESTINATION examples/pixel_level_classification )
