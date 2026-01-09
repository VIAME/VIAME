set( EXAMPLE_INSTALL_FILES
     README.rst
     calibration_matrices.npz
     cam1_images.txt
     cam2_images.txt )

if( WIN32 )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       calibrate_cameras.bat
       measure_via_gmm_oriented_boxes.bat
       measure_via_default_fish.bat
       measure_over_manual_annotations.bat
       compute_depth_maps.bat )
else()
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       calibrate_cameras.sh
       measure_via_gmm_oriented_boxes.sh
       measure_via_default_fish.sh
       measure_over_manual_annotations.sh
       gmm_standalone_tool.sh
       compute_depth_maps.sh )
endif()

install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/size_measurement )
