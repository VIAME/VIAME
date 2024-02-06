###
# Conversion scripts for type conversion
##

set( example_files
     README.rst )

install( FILES       ${example_files}
         DESTINATION examples/detection_file_conversions )
install( DIRECTORY   scripts
         DESTINATION examples/detection_file_conversions )
