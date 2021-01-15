###
# Application level pipelines
##

set( EXAMPLE_INSTALL_FILES
     input_image_list_habcam.txt
     input_image_list_small_set.txt )

if( WIN32 )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       run_fish_without_motion.bat
       run_generic_proposals.bat )
else()
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       run_fish_without_motion.sh
       run_generic_proposals.sh )
endif()

if( VIAME_DOWNLOAD_MODELS-ARCTIC-SEAL )

  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       input_image_list_seal_eo.txt
       input_image_list_seal_ir.txt )

  if( VIAME_ENABLE_TENSORFLOW )
    if( WIN32 )
      set( EXAMPLE_INSTALL_FILES
           ${EXAMPLE_INSTALL_FILES}
           run_arctic_seal_eo_tf.bat
           run_arctic_seal_ir_tf.bat )
    else()
      set( EXAMPLE_INSTALL_FILES
           ${EXAMPLE_INSTALL_FILES}
           run_arctic_seal_eo_tf.sh
           run_arctic_seal_ir_tf.sh )
    endif()
  endif()

  if( VIAME_ENABLE_DARKNET )
    if( WIN32 )
      set( EXAMPLE_INSTALL_FILES
           ${EXAMPLE_INSTALL_FILES}
           run_arctic_seal_fusion_yolo.bat
           run_arctic_seal_eo_yolo.bat
           run_arctic_seal_ir_yolo.bat )
    else()
      set( EXAMPLE_INSTALL_FILES
           ${EXAMPLE_INSTALL_FILES}
           run_arctic_seal_fusion_yolo.sh
           run_arctic_seal_eo_yolo.sh
           run_arctic_seal_ir_yolo.sh )
    endif()
  endif()
endif()

###
# Install pipelines to example directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/object_detection )
