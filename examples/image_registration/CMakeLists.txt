###
# Image Enhancement and Filtering Example
##
set( EXAMPLE_INSTALL_FILES
     README.rst )

if( VIAME_ENABLE_ITK )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
      ${EXAMPLE_INSTALL_FILES}
      input_optical_images.txt
      input_thermal_images.txt
      register_optical_and_thermal.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
      ${EXAMPLE_INSTALL_FILES}
      input_optical_images.txt
      input_thermal_images.txt
      register_optical_and_thermal.sh )
  endif()
endif()

###
# Install pipelines to example directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/image_registration )
