###
# Image Enhancement and Filtering Example
##
set( EXAMPLE_INSTALL_FILES
     README.rst )

if( VIAME_ENABLE_OPENCV )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         input_list_raw_images.txt
         input_list_rgb_images.txt
         input_list_16bit_images.txt
         debayer_and_enhance.bat
         enhance.bat
         normalize_16bit.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         input_list_raw_images.txt
         input_list_rgb_images.txt
         input_list_16bit_images.txt
         debayer_and_enhance.sh
         enhance.sh
         normalize_16bit.sh )
  endif()
endif()

###
# Install pipelines to example directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/image_enhancement )
