###
# Application level pipelines
##
set( VIAME_HELLO_WORLD_PIPELINES
     hello_world_detector.pipe )

###
# Install pipelines to example directory
##
install( FILES       ${VIAME_HELLO_WORLD_PIPELINES}
         DESTINATION examples/hello_world_pipeline )
