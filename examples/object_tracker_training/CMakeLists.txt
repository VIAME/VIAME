set( EXAMPLE_INSTALL_FILES
     README.rst )

# ByteTrack and OC-SORT trainers are parameter-estimation only (no ML)
# and are always available when PyTorch is enabled
if( WIN32 )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       train_bytetrack.bat
       train_ocsort.bat
       train_adaptive.bat )
else()
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       train_bytetrack.sh
       train_ocsort.sh
       train_adaptive.sh )
endif()

if( VIAME_ENABLE_PYTORCH )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_deepsort.bat
         train_botsort.bat
         train_srnn.bat
         train_siammask.bat
         train_st_tracker_viame_csv.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         train_deepsort.sh
         train_botsort.sh
         train_srnn.sh
         train_siammask.sh
         train_st_tracker_viame_csv.sh )
  endif()
endif()

###
# Install pipelines to example directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/object_tracker_training )
