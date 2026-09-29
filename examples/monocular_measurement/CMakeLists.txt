###
# Monocular measurement examples
##

set( EXAMPLE_INSTALL_FILES
     README.rst
     input_image_list_habcam.txt )

if( WIN32 )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       calibrate_single_camera.bat
       measure_via_habcam_metadata.bat )
else()
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       calibrate_single_camera.sh
       measure_via_habcam_metadata.sh )
endif()

install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/monocular_measurement )
