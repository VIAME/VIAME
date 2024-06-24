###
# Build Example Files
##

set( example_files
  README.rst )

if( WIN32 )
  set( example_files
       ${example_files} )
else()
  set( example_files
       ${example_files}
       example.sh )
endif()

install( FILES       ${example_files}
         DESTINATION examples/building_from_source )
