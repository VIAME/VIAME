###
# Example files for each annotation format
##

set( EXAMPLE_INSTALL_FILES
     README.rst
     example.viame.csv
     example.coco.json
     example.dive.json )

install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/detection_file_formats )
