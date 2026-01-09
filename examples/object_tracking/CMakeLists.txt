###
# Application level pipelines
##

set( EXAMPLE_INSTALL_FILES
     README.rst
     input_list.txt )

if( WIN32 )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       run_fish_tracker.bat
       run_generic_tracker.bat
       run_simple_tracker.bat
       run_track_viewer.bat )
elseif( APPLE )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       run_fish_tracker.sh
       run_generic_tracker.sh
       run_simple_tracker.sh
       run_track_viewer.mac.sh )
else()
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       run_fish_tracker.sh
       run_generic_tracker.sh
       run_simple_tracker.sh
       run_track_viewer.sh )
endif()

if( VIAME_ENABLE_PYTORCH-SIAMMASK )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         bulk_run_user_init_tracking.bat
         run_user_init_tracker.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         bulk_run_user_init_tracking.sh
         run_user_init_tracker.sh )
  endif()
endif()

###
# Install pipelines to example directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/object_tracking )
