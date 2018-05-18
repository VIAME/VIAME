set( EXAMPLE_INSTALL_FILES
     README.rst
     calibration.npz
     cam1_images.txt
     cam2_images.txt )

if( WIN32 )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       run_measurer.bat )
else()
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       run_measurer.sh )
endif()

install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/measurement_using_stereo )
