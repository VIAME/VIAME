###
# Conversion scripts for type conversion
##

set( EXAMPLE_INSTALL_FILES
     README.rst )

if( WIN32 )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       bulk_convert_gt_plus_data.bat
       bulk_convert_gt_only.bat )
else()
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       bulk_convert_gt_plus_data.sh
       bulk_convert_gt_only.sh )
endif()

install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/detection_file_conversions )
install( DIRECTORY   standalone_utils
         DESTINATION examples/detection_file_conversions )
