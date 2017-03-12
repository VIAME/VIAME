###
# Application level pipelines
##

set( EXAMPLE_INSTALL_FILES
     input_list.txt )

if( VIAME_ENABLE_YOLO )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       yolo_v2_detector.pipe )
endif()

if( VIAME_ENABLE_SCALLOP_TK )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       scallop_tk_detector.pipe )
endif()

if( VIAME_ENABLE_SCALLOP_TK AND VIAME_ENABLE_YOLO AND VIAME_ENABLE_FASTER_RCNN )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       multi_cnn_detector.pipe )
endif()

if( VIAME_ENABLE_MATLAB AND VIAME_ENABLE_UW_PREDICTOR )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       camtrawl_pipeline.pipe )
endif()

###
# Install pipelines to example directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/detector_pipelines )
