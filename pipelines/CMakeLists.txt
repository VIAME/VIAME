###
# Application level pipelines
##
set( VIAME_PIPELINES )

if( VIAME_ENABLE_SCALLOP_TK )
  set( VIAME_PIPELINES
       ${VIAME_PIPELINES}
       scallop_tk_detector.pipe )
endif()

###
# Install pipelines to pipeline directory
##
install( FILES ${VIAME_PIPELINES} DESTINATION pipelines )
