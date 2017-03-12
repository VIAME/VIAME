###
# Application level pipelines
##
set( HELLO_WORLD_EX_SOURCE
     hello_world_detector.pipe )

if( WIN32 )
  set( HELLO_WORLD_EX_SOURCE
       ${HELLO_WORLD_EX_SOURCE}
       run_example.bat )
else()
  set( HELLO_WORLD_EX_SOURCE
       ${HELLO_WORLD_EX_SOURCE}
       run_example.sh )
endif()

###
# Install pipelines to example directory
##
install( FILES       ${HELLO_WORLD_EX_SOURCE}
         DESTINATION examples/hello_world_pipeline )
