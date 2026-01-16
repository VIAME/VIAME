
set( ADD_ON_PIPELINE_FILES )

if( VIAME_DOWNLOAD_MODELS-ARCTIC-SEAL )
  if( NOT VIAME_ENABLE_DARKNET )
    message( FATAL_ERROR "Arctic seal models require darknet" )
  endif()

  DownloadAndExtract(
    https://data.kitware.com/api/v1/item/5d6edf99d35580e6dcddbc3b/download
    1a122a8a5ab5e59ca8a149954da8a252
    ${VIAME_DOWNLOAD_DIR}/models-arctic_seal_transforms-v0.2.tar.gz
    ${CMAKE_CURRENT_SOURCE_DIR} )
  DownloadAndExtract(
    https://data.kitware.com/api/v1/item/5cbb84d08d777f072b53ab46/download
    5d92853d93e758260dfefbf12ab70ab5
    ${VIAME_DOWNLOAD_DIR}/models-arctic_seal_yolo-v1.3.tar.gz
    ${CMAKE_CURRENT_SOURCE_DIR} )

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
    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5cb74eeb8d777f072b40f904/download
      5af19ef3d8c29d3726674421912a44dc
      ${VIAME_DOWNLOAD_DIR}/models-arctic_seal_tf-v1.1.1.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR} )

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

if( VIAME_DOWNLOAD_MODELS-IFREMER )

endif()

if( VIAME_ENABLE_PYTORCH-LEARN )
  if( NOT VIAME_ENABLE_PYTORCH )
    message( FATAL_ERROR "Learn models require pytorch" )
  endif()

  set( ADD_ON_PIPELINE_FILES ${ADD_ON_PIPELINE_FILES}
    learn/detector_maskcut_unspvr_segmenter.pipe )

  if( VIAME_DOWNLOAD_MODELS-CONVNEXT )
    DownloadAndInstallAddOn(
      https://viame.kitware.com/girder/api/v1/item/64e2c485ef791ec92a7221b2/download
      485daf5a1a79ba73860334fbc36c58f5
      ${VIAME_DOWNLOAD_DIR}/VIAME-ConvNeXT-Models-v1.1.zip )

    set( ADD_ON_PIPELINE_FILES ${ADD_ON_PIPELINE_FILES}
      learn/detector_convnext_baseline_xl.pipe
      learn/train_detector_mmdet_convnext_cascade.conf )
  endif()
  if( VIAME_DOWNLOAD_MODELS-POOLCAR )
    DownloadAndInstallAddOn(
      https://viame.kitware.com/girder/api/v1/item/649cde32fb1572b3599ec620/download
      60a5249405577be31cdf39cce06c04f6
      ${VIAME_DOWNLOAD_DIR}/VIAME-PoolCar-Example-v1.0.zip )

    set( ADD_ON_PIPELINE_FILES ${ADD_ON_PIPELINE_FILES}
      learn/detector_cutler_10_shot_poolcar.pipe )
  endif()
endif()

if( VIAME_DOWNLOAD_MODELS-HABCAM )
  if( NOT VIAME_ENABLE_PYTORCH )
    message( FATAL_ERROR "HabCam models require pytorch" )
  endif()

  if( WIN32 )
    DownloadAndInstallAddOn(
      https://viame.kitware.com/api/v1/item/63e1e605ef73c03d10a77531/download
      32ec45d7095f7c2f6a42ebbb11b2b212
      ${VIAME_DOWNLOAD_DIR}/VIAME-HabCam-Models-v2.0.zip )
  else()
    DownloadAndInstallAddOn(
      https://viame.kitware.com/api/v1/item/63e5b26163ed1fd08e734ba6/download
      ef285bc3d055aaa2de7bcd0ca7dd5e45
      ${VIAME_DOWNLOAD_DIR}/VIAME-HabCam-Models-v2.1.zip )
  endif()
endif()

if( VIAME_DOWNLOAD_MODELS-MOUSS )
  if( NOT VIAME_ENABLE_PYTORCH )
    message( FATAL_ERROR "MOUSS models require pytorch" )
  endif()

  DownloadAndInstallAddOn(
    https://viame.kitware.com/api/v1/item/627b3282c4da86e2cd3abb5d/download
    87b9f9bf5088dbce46eb73e03f1597e2
    ${VIAME_DOWNLOAD_DIR}/VIAME-MOUSS-Deep7-v2.1.zip )
endif()

if( VIAME_DOWNLOAD_MODELS-SEA-LION )
  if( NOT VIAME_ENABLE_PYTORCH )
    message( FATAL_ERROR "Sea Lion models require pytorch" )
  endif()

  DownloadAndInstallAddOn(
    https://viame.kitware.com/api/v1/item/64e391660ee78064c384dbb9/download
    384158dba64b53bed5c6f800b3c6d869
    ${VIAME_DOWNLOAD_DIR}/VIAME-Sea-Lion-Models-v2.11.zip )
endif()

if( VIAME_DOWNLOAD_MODELS-UW-CLFR )
  # Install UW Fish Hierarchical SVM classifier models
  set( UW_FISH_MODEL_FILES
    uw-fish-classifier/Model_SVM.xml
    uw-fish-classifier/speciesSVM_1.xml
    uw-fish-classifier/speciesSVM_2.xml
    uw-fish-classifier/speciesSVM_3.xml
    uw-fish-classifier/speciesSVM_4.xml
    uw-fish-classifier/speciesSVM_5.xml
    uw-fish-classifier/speciesSVM_6.xml
    uw-fish-classifier/speciesSVM_7.xml
    uw-fish-classifier/speciesSVM_8.xml
    uw-fish-classifier/speciesSVM_9.xml
    uw-fish-classifier/speciesSVM_10.xml
    uw-fish-classifier/speciesSVM_11.xml
    uw-fish-classifier/speciesSVM_12.xml
    uw-fish-classifier/speciesSVM_13.xml
    uw-fish-classifier/speciesSVM_14.xml
    uw-fish-classifier/speciesSVM_15.xml
  )

  foreach( MODEL_FILE ${UW_FISH_MODEL_FILES} )
    install( FILES ${MODEL_FILE}
             DESTINATION configs/models/uw-fish-classifier )
  endforeach()

  set( ADD_ON_PIPELINE_FILES ${ADD_ON_PIPELINE_FILES}
    uw-fish-classifier/detector_matlab_camtrawl.pipe )
endif()

if( ADD_ON_PIPELINE_FILES )
  foreach( ITEM ${ADD_ON_PIPELINE_FILES} )
    string( REGEX REPLACE "^[^/]+/" "" ITEM_NFF "${ITEM}" )
    get_filename_component( ITEM_PATH_NFF "${ITEM_NFF}" DIRECTORY )
    install( FILES ${ITEM}
             DESTINATION configs/pipelines/${ITEM_PATH_NFF} )
  endforeach()
endif()
