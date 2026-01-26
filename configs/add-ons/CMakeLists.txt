
# Validate that all required dependencies are enabled for selected addons
ValidateAddonDependencies()

# Process downloads and install models/transforms for all enabled addons
ProcessAllAddonModels()

# Pipeline files to install from source tree
set( ADD_ON_PIPELINE_FILES )

if( VIAME_DOWNLOAD_MODELS-ARCTIC-SEAL )
  set( ADD_ON_PIPELINE_FILES ${ADD_ON_PIPELINE_FILES}
    artic-seal/common_arctic_seal_eo_yolo_detector.pipe
    artic-seal/common_arctic_seal_eo_tiny_yolo_detector.pipe
    artic-seal/common_arctic_seal_ir_yolo_detector.pipe
    artic-seal/common_arctic_seal_ir_tiny_yolo_detector.pipe
    artic-seal/detector_arctic_seal_fusion_yolo.pipe
    artic-seal/detector_arctic_seal_eo_yolo.pipe
    artic-seal/detector_arctic_seal_ir_yolo.pipe
  )

  set( ADD_ON_PIPELINE_FILES ${ADD_ON_PIPELINE_FILES}
    artic-seal/embedded_single_stream/arctic_seal_eo_tiny_yolo_detector.pipe
    artic-seal/embedded_single_stream/arctic_seal_eo_tiny_yolo_tracker.pipe
    artic-seal/embedded_single_stream/arctic_seal_eo_tiny_yolo_tracker_alt2.pipe
    artic-seal/embedded_single_stream/arctic_seal_eo_tiny_yolo_tracker_alt3.pipe
    artic-seal/embedded_single_stream/arctic_seal_eo_yolo_detector.pipe
    artic-seal/embedded_single_stream/arctic_seal_ir_tiny_yolo_detector.pipe
    artic-seal/embedded_single_stream/arctic_seal_ir_yolo_detector.pipe
  )

  set( ADD_ON_PIPELINE_FILES ${ADD_ON_PIPELINE_FILES}
    artic-seal/embedded_dual_stream/arctic_seal_tf_eo_only.pipe
    artic-seal/embedded_dual_stream/arctic_seal_tf_ir_only.pipe
    artic-seal/embedded_dual_stream/arctic_seal_tf_ir_to_eo_frame_trigger.pipe
    artic-seal/embedded_dual_stream/arctic_seal_tiny_yolo_eo_only.pipe
    artic-seal/embedded_dual_stream/arctic_seal_tiny_yolo_eo_tracker.pipe
    artic-seal/embedded_dual_stream/arctic_seal_tiny_yolo_eo_tracker_alt.pipe
    artic-seal/embedded_dual_stream/arctic_seal_yolo_eo_ir_early_fusion.pipe
    artic-seal/embedded_dual_stream/arctic_seal_yolo_eo_only.pipe
    artic-seal/embedded_dual_stream/arctic_seal_yolo_ir_only.pipe
    artic-seal/embedded_dual_stream/arctic_seal_yolo_ir_to_eo_frame_trigger.pipe
    artic-seal/embedded_dual_stream/arctic_seal_yolo_ir_to_eo_region_trigger.pipe
    artic-seal/embedded_dual_stream/arctic_seal_yolo_ir_to_tiny_eo_region_tigger.pipe
  )

  if( VIAME_ENABLE_TENSORFLOW )
    set( ADD_ON_PIPELINE_FILES ${ADD_ON_PIPELINE_FILES}
      artic-seal/common_arctic_seal_eo_tf_detector.pipe
      artic-seal/common_arctic_seal_ir_tf_detector.pipe
      artic-seal/detector_arctic_seal_eo_tf.pipe
      artic-seal/detector_arctic_seal_ir_tf.pipe
    )

    set( ADD_ON_PIPELINE_FILES ${ADD_ON_PIPELINE_FILES}
      artic-seal/embedded_single_stream/arctic_seal_eo_tf_detector.pipe
      artic-seal/embedded_single_stream/arctic_seal_ir_tf_detector.pipe )
  endif()
endif()

if( VIAME_ENABLE_PYTORCH-LEARN )
  set( ADD_ON_PIPELINE_FILES ${ADD_ON_PIPELINE_FILES}
    learn/detector_maskcut_unspvr_segmenter.pipe )

  if( VIAME_DOWNLOAD_MODELS-CONVNEXT )
    set( ADD_ON_PIPELINE_FILES ${ADD_ON_PIPELINE_FILES}
      learn/detector_convnext_baseline_xl.pipe
      learn/train_detector_mmdet_convnext_cascade.conf )
  endif()
endif()

if( ADD_ON_PIPELINE_FILES )
  foreach( ITEM ${ADD_ON_PIPELINE_FILES} )
    string( REGEX REPLACE "^[^/]+/" "" ITEM_NFF "${ITEM}" )
    get_filename_component( ITEM_PATH_NFF "${ITEM_NFF}" DIRECTORY )
    install( FILES ${ITEM}
             DESTINATION configs/pipelines/${ITEM_PATH_NFF} )
  endforeach()
endif()
