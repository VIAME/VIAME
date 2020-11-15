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
       run_mouss_tracker.bat
       run_pysot_tracker.bat
       run_simple_tracker.bat
       run_track_viewer.bat )
elseif( APPLE )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       run_fish_tracker.sh
       run_generic_tracker.sh
       run_mouss_tracker.sh
       run_pysot_tracker.sh
       run_simple_tracker.sh
       run_track_viewer.mac.sh )
else()
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       run_default_tracker.sh
       run_generic_tracker.sh
       run_mouss_tracker.sh
       run_pysot_tracker.sh
       run_simple_tracker.sh
       run_track_viewer.sh )
endif()

###
# Install pipelines to example directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/object_tracking )
