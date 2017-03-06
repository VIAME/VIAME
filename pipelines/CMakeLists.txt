###
# Application level pipelines
##
set( VIAME_PIPELINES
     hello_world_detector.pipe )

if( VIAME_ENABLE_SCALLOP_TK )
  set( VIAME_PIPELINES
       ${VIAME_PIPELINES}
       scallop_tk_detector.pipe )
endif()

if( VIAME_ENABLE_MATLAB AND VIAME_ENABLE_UW_PREDICTOR )
  set( VIAME_PIPELINES
       ${VIAME_PIPELINES}
       camtrawl_pipeline.pipe )
endif()

###
# Install pipelines to pipeline directory
##
install( FILES ${VIAME_PIPELINES} DESTINATION pipelines )
