###
# Image Enhancement and Filtering Example
##
set( EXAMPLE_INSTALL_FILES
     README.rst )

if( VIAME_ENABLE_OPENCV )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         input_list.txt
         debayer_and_enhance.bat
         enhance.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         input_list.txt
         debayer_and_enhance.sh
         enhance.sh )
  endif()
endif()

###
# Install pipelines to example directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/image_enhancement )
