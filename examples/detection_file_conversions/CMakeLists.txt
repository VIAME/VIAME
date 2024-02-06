###
# Conversion scripts for type conversion
##

set( EXAMPLE_INSTALL_FILES
     README.rst )

if( WIN32 )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       bulk_convert_using_pipe.bat )
else()
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       bulk_convert_using_pipe.sh )
endif()

install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/detection_file_conversions )
install( DIRECTORY   scripts
         DESTINATION examples/detection_file_conversions )
