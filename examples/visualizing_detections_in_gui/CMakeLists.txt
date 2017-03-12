###
# GUI Project File Examples
##

set( example_files
  display_annotation.pipe )

if( WIN32 )
  set( HELLO_WORLD_EX_SOURCE
       ${HELLO_WORLD_EX_SOURCE}
       run_example.bat )
else()
  set( HELLO_WORLD_EX_SOURCE
       ${HELLO_WORLD_EX_SOURCE}
       run_example.sh )
endif()

install( FILES       ${example_files}
         DESTINATION examples/visualizing_detections_in_gui )
