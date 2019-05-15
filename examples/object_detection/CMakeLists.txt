###
# Application level pipelines
##

set( EXAMPLE_INSTALL_FILES
     input_image_list_habcam.txt
     input_image_list_small_set.txt )

if( VIAME_ENABLE_PYTORCH )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         run_default.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         run_default.sh )
  endif()
endif()

if( VIAME_ENABLE_YOLO )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         run_habcam.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         run_habcam.sh )
  endif()
endif()

if( VIAME_ENABLE_MATLAB )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         run_lanl.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         run_lanl.sh )
  endif()
endif()

if( VIAME_ENABLE_SCALLOP_TK )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         run_scallop_tk.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         run_scallop_tk.sh )
  endif()
endif()

if( VIAME_ENABLE_TENSORFLOW AND VIAME_DOWNLOAD_MODELS-ARCTIC-SEAL )
  if( NOT WIN32 )
    set( EXAMPLE_INSTALL_FILES
         ${EXAMPLE_INSTALL_FILES}
         input_image_list_seal_eo.txt
         input_image_list_seal_ir.txt
         run_arctic_seal_fusion_yolo.sh
         run_arctic_seal_eo_tf.sh
         run_arctic_seal_ir_tf.sh )
  endif()
endif()

###
# Install pipelines to example directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/object_detection )
