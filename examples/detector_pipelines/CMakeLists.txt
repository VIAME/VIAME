###
# Application level pipelines
##
set( VIAME_DETECTOR_PIPELINES
     hello_world_detector.pipe )

if( VIAME_ENABLE_SCALLOP_TK )
  set( VIAME_DETECTOR_PIPELINES
       ${VIAME_DETECTOR_PIPELINES}
       scallop_tk_detector.pipe )
endif()

if( VIAME_ENABLE_MATLAB AND VIAME_ENABLE_UW_PREDICTOR )
  set( VIAME_DETECTOR_PIPELINES
       ${VIAME_DETECTOR_PIPELINES}
       camtrawl_pipeline.pipe )
endif()

###
# Install pipelines to example directory
##
install( FILES       ${VIAME_DETECTOR_PIPELINES}
         DESTINATION examples/detector_pipelines )
