set( EXAMPLE_INSTALL_FILES
     README.rst
     calibration_matrices.npz
     cam1_images.txt
     cam2_images.txt )

if( WIN32 )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       run_measurer.bat
       compute_depth_maps.bat )
else()
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       run_measurer.sh
       standalone_with_disp.sh
       compute_depth_maps.sh )
endif()

install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/size_measurement )
