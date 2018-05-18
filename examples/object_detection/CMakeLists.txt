###
# Application level pipelines
##

set( EXAMPLE_INSTALL_FILES
     input_list.txt
     habcam_list.txt )

if( VIAME_ENABLE_YOLO )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         run_yolo.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         run_yolo.sh )
  endif()

  if( NOT WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         run_habcam.sh )
  endif()
endif()


if( VIAME_ENABLE_MATLAB )
  if( NOT WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         run_lanl.sh )
  endif()
endif()

###
# Install pipelines to example directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/object_detection )
install( DIRECTORY   models
         DESTINATION examples/object_detection )
install( FILES       output/example_detections.kw18
                     output/example_tot_scores.txt
                     output/yolo_v2_detections.prj
         DESTINATION examples/object_detection/output )
