###
# Download and install enabled model files
##
if( VIAME_DOWNLOAD_MODELS )

  # GUI background context view
  if( VIAME_ENABLE_VIVIA )
    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5afe6c018d777f15ebe1d6f2/download
      7e8800cb11a1a491140e7f58e9053030
      ${VIAME_DOWNLOAD_DIR}/models-blue_marble_context-v1.0.0.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}/gui-params )
  endif()

  # Default fish detection model
  if( VIAME_DOWNLOAD_MODELS-FISH )
    if( VIAME_ENABLE_PYTORCH-NETHARN )
      DownloadAndExtract(
        https://data.kitware.com/api/v1/item/626e0c9e4acac99f425acf1e/download
        4971e8b78003d51796a07f864dbf5591
        ${VIAME_DOWNLOAD_DIR}/models-netharn_fish-v1.5.zip
        ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
    elseif( VIAME_ENABLE_PYTORCH-MMDET )
      DownloadAndExtract(
        https://data.kitware.com/api/v1/item/5fb196f250a41e3d1956a407/download
        c6e0cdf1e903dbe3e261977378a97223
        ${VIAME_DOWNLOAD_DIR}/models-mmdet_default-v1.4.tar.gz
        ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
    elseif( VIAME_ENABLE_DARKNET )
      DownloadAndInstallAddOn(
        https://viame.kitware.com/api/v1/item/6172423562e0e5717e22cc21/download
        5b8a947d4d9de3ecc7acd8e729652ada
        ${VIAME_DOWNLOAD_DIR}/VIAME-Default-CPU-Detectors-v1.0.zip )
    endif()
  endif()

  # Generic object detection model
  if( VIAME_DOWNLOAD_MODELS-GENERIC )
    if( VIAME_ENABLE_DARKNET )
      DownloadAndExtract(
        https://data.kitware.com/api/v1/item/626e05b54acac99f425acc46/download
        72fe8f6cc1653b350154e32d03efd6c0
        ${VIAME_DOWNLOAD_DIR}/models-yolo_generic-v1.5.zip
        ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
    elseif( VIAME_ENABLE_PYTORCH-NETHARN )
      DownloadAndExtract(
        https://data.kitware.com/api/v1/item/62682a514acac99f420f6883/download
        c71f276ac893502803504dad10150cc8
        ${VIAME_DOWNLOAD_DIR}/models-netharn_generic-v1.4.zip
        ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
    elseif( VIAME_ENABLE_PYTORCH-MMDET )
      DownloadAndExtract(
        https://data.kitware.com/api/v1/item/5fb1922650a41e3d1956a389/download
        bf2df19944c55e8a7dc757520c607893
        ${VIAME_DOWNLOAD_DIR}/models-mmdet_generic-v1.4.tar.gz
        ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
    endif()
  endif()

  # Motion detection model
  if( VIAME_DOWNLOAD_MODELS-MOTION )
    if( VIAME_ENABLE_PYTORCH-NETHARN )
      DownloadAndInstallAddOn(
        https://data.kitware.com/api/v1/item/601b00d02fa25629b9391ad6/download
        6524dd85334d569d30fa9ca7eb570517
        ${VIAME_DOWNLOAD_DIR}/VIAME-Motion-Models-v1.0.zip )
    endif()
  endif()

  # Other PyTorch detection, tracking, and training (seed) models
  if( VIAME_ENABLE_PYTORCH )

    # Seed models for model training
    if( VIAME_ENABLE_PYTORCH-NETHARN OR VIAME_ENABLE_PYTORCH-MMDET )
      DownloadAndExtract(
        https://data.kitware.com/api/v1/item/5fb1922d50a41e3d1956a396/download
        d1befaf3b64c185f40a29ed2fca87929
        ${VIAME_DOWNLOAD_DIR}/models-mmdet_primitives-v1.4.tar.gz
        ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
    endif()

    # Default annotation assist tracking models
    if( VIAME_ENABLE_PYTORCH-PYSOT )
      DownloadAndExtract(
        https://viame.kitware.com/api/v1/item/61e4e0ebe90592181da06a79/download
        39c2a4de745edbed0452f8e951c57537
        ${VIAME_DOWNLOAD_DIR}/models-pysot_tracking_models-v1.2.zip
        ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
    endif()

    if( VIAME_ENABLE_PYTORCH-MDNET )
      DownloadAndExtract(
        https://data.kitware.com/api/v1/item/5e12b04aaf2e2eed35d42979/download
        8f6a709478bbd69e4d97fc34ac9b6003
        ${VIAME_DOWNLOAD_DIR}/models-mdnet_tracking_models-v1.0.0.tar.gz
        ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
    endif()

    # Default multi-target tracking models
    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5facb1bb50a41e3d194d2af6/download
      eb66d389e7fd58529b5cd4b00d6b5058
      ${VIAME_DOWNLOAD_DIR}/models-pytorch_tracking-v1.2.0.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )

    # Default descriptor models
    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5bad56178d777f06b908c47c/download
      3fe6c43e8bf6ab49cb8f6687f38bdbd6
      ${VIAME_DOWNLOAD_DIR}/models-pytorch_descriptors-v1.1.0.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
  endif()

  # Darknet training seed models
  if( VIAME_ENABLE_DARKNET )
    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/602b56c72fa25629b9788909/download
      a4a7805499527c58b5165f0901fc0ec3
      ${VIAME_DOWNLOAD_DIR}/models-yolo_v4_seed_model-v1.0.0.zip
      ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
  endif()

  # Caffe generic descriptor
  if( VIAME_ENABLE_CAFFE )
    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5afe6e1e8d777f15ebe1d6fc/download
      2cdbe7ca51b49b12e59911aadec20e98
      ${VIAME_DOWNLOAD_DIR}/models-default_caffe_descriptor-v1.0.0.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
  endif()

  # Mixed project specific models and add-ons
  if( VIAME_DOWNLOAD_MODELS-HABCAM )
    DownloadAndInstallAddOn(
      https://viame.kitware.com/api/v1/item/613ed525a28c6cb53f3a0601/download
      6582b5dd9485727dd40e5582defc0389
      ${VIAME_DOWNLOAD_DIR}/VIAME-HabCam-Models-v1.8.zip )
  endif()

  if( VIAME_DOWNLOAD_MODELS-MOUSS )
    DownloadAndInstallAddOn(
      https://viame.kitware.com/api/v1/item/616d9bf7b53494023680648c/download
      c1200d203ba07eaeddf03633d0eb1bf9
      ${VIAME_DOWNLOAD_DIR}/VIAME-MOUSS-Deep7-v2.1.zip )
  endif()

  if( VIAME_DOWNLOAD_MODELS-ARCTIC-SEAL )
    if( NOT VIAME_ENABLE_DARKNET )
      message( FATAL_ERROR "Arctic seal models require darknet" )
    endif()
    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5d6edf99d35580e6dcddbc3b/download
      1a122a8a5ab5e59ca8a149954da8a252
      ${VIAME_DOWNLOAD_DIR}/models-arctic_seal_transforms-v0.2.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5cbb84d08d777f072b53ab46/download
      5d92853d93e758260dfefbf12ab70ab5
      ${VIAME_DOWNLOAD_DIR}/models-arctic_seal_yolo-v1.3.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
    if( VIAME_ENABLE_TENSORFLOW )
      DownloadAndExtract(
        https://data.kitware.com/api/v1/item/5cb74eeb8d777f072b40f904/download
        5af19ef3d8c29d3726674421912a44dc
        ${VIAME_DOWNLOAD_DIR}/models-arctic_seal_tf-v1.1.1.tar.gz
        ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
    endif()
  endif()

  if( VIAME_DOWNLOAD_MODELS-SEA-LION )
    if( NOT VIAME_ENABLE_PYTORCH )
      message( FATAL_ERROR "Sea Lion models require pytorch" )
    endif()
    DownloadAndInstallAddOn(
      https://data.kitware.com/api/v1/item/6011b1bc2fa25629b91a5e3c/download
      cccd42702eead4148de7d26c53bb3718
      ${VIAME_DOWNLOAD_DIR}/VIAME-Sea-Lion-Models-v1.3.zip )
  endif()
endif()

###
# All subdirectories for this folder installed as-is
##
install( DIRECTORY gui-params
         DESTINATION configs )

if( WIN32 )
  install( DIRECTORY prj-windows/
           DESTINATION configs/prj-template )
else()
  install( DIRECTORY prj-linux/
           DESTINATION configs/prj-template )
endif()

###
# All config files manually-included by default or based on options
##
set( CORE_PIPELINE_FILES
  pipelines/classify_detections_svm.pipe
  pipelines/common_default_initializer.pipe
  pipelines/common_default_input.pipe
  pipelines/common_default_input_with_downsampler.pipe
  pipelines/common_fish_detector.pipe
  pipelines/common_fish_detector_with_filter.pipe
  pipelines/common_generic_detector.pipe
  pipelines/common_generic_detector_with_filter.pipe
  pipelines/common_image_stabilizer.pipe
  pipelines/common_stabilized_iou_tracker.pipe
  pipelines/database_apply_svm_models.pipe
  pipelines/detector_camtrawl.pipe
  pipelines/detector_extract_chips.pipe
  pipelines/detector_fish_without_motion.pipe
  pipelines/detector_generic_proposals.pipe
  pipelines/detector_project_folder.pipe
  pipelines/detector_project_folder_left.pipe
  pipelines/detector_simple_hough.pipe
  pipelines/detector_svm_models.pipe
  pipelines/display_annotations.pipe
  pipelines/draw_detections_on_images.pipe
  pipelines/extract_chips_from_detections.pipe
  pipelines/filter_debayer_and_enhance.pipe
  pipelines/filter_enhance.pipe
  pipelines/filter_extract_frames.pipe
  pipelines/filter_extract_left.pipe
  pipelines/filter_split_and_debayer.pipe
  pipelines/filter_stereo_depth_map.pipe
  pipelines/filter_to_kwa.pipe
  pipelines/frame_classifier_project_folder.pipe
  pipelines/frame_classifier_svm.pipe
  pipelines/index_default.pipe
  pipelines/index_default.svm.pipe
  pipelines/index_default.trk.pipe
  pipelines/index_existing.pipe
  pipelines/index_fish.svm.pipe
  pipelines/index_frame.pipe
  pipelines/index_frame.svm.pipe
  pipelines/index_frame.svm.annot_only.pipe
  pipelines/measurement_default.trk.pipe
  pipelines/measurement_gmm_only.pipe
  pipelines/query_image_exemplar.cfe.pipe
  pipelines/query_image_exemplar.pipe
  pipelines/query_retrieval_and_iqr.cfe.pipe
  pipelines/query_retrieval_and_iqr.pipe
  pipelines/query_video_exemplar.pipe
  pipelines/sql_init_table.sql
  pipelines/tracker_fish.pipe
  pipelines/tracker_generic.pipe
  pipelines/tracker_project_folder.pipe
  pipelines/tracker_stabilized_iou.pipe
  pipelines/tracker_svm_models.pipe
  pipelines/train_aug_add_color_freq.pipe
  pipelines/train_aug_add_double_motion.pipe
  pipelines/train_aug_add_motion_and_color_freq.pipe
  pipelines/train_aug_add_motion.pipe
  pipelines/train_aug_enhance_and_add_motion.pipe
  pipelines/train_aug_hue_shifting_only.pipe
  pipelines/train_aug_intensity_color_freq_motion.pipe
  pipelines/train_aug_intensity_hue_motion.pipe
  pipelines/train_aug_motion_only.pipe
  pipelines/train_aug_split_and_stereo.pipe
  pipelines/train_aug_split.pipe
  pipelines/train_aug_warp_ir_to_eo.pipe
  pipelines/utility_add_head_tail_keypoints.pipe
  pipelines/utility_add_segmentations_watershed.pipe
  pipelines/utility_empty_frame_lbls_1fr.pipe
  pipelines/utility_empty_frame_lbls_10fr.pipe
  pipelines/utility_empty_frame_lbls_100fr.pipe
  pipelines/utility_empty_frame_lbls_1000fr.pipe
  pipelines/utility_max_10_points_per_poly.pipe
  pipelines/utility_max_25_points_per_poly.pipe )

# Configs which differ based on CPU vs GPU
if( VIAME_ENABLE_CUDA )
  set( CORE_PIPELINE_FILES ${CORE_PIPELINE_FILES}
    pipelines/common_default_descriptor.pipe
    pipelines/common_default_tracker.pipe
    pipelines/query_augment_image.pipe
    pipelines/tracker_fish.sfd.pipe )
else()
  install( FILES pipelines/common_default_descriptor.cpu.pipe
    DESTINATION configs/pipelines
    RENAME common_default_descriptor.pipe )
  install( FILES pipelines/common_default_tracker.cpu.pipe
    DESTINATION configs/pipelines
    RENAME common_default_tracker.pipe )
  install( FILES pipelines/query_augment_image.cpu.pipe
    DESTINATION configs/pipelines
    RENAME query_augment_image.pipe )
  install( FILES pipelines/tracker_fish.sfd.cpu.pipe
    DESTINATION configs/pipelines
    RENAME tracker_fish.sfd.pipe )
endif()

# Pytorch specific configs
if( VIAME_ENABLE_PYTORCH )
  set( CORE_PIPELINE_FILES ${CORE_PIPELINE_FILES}
    pipelines/train_detector_default.viame_csv.conf
    pipelines/train_frame_classifier_default.viame_csv.conf
    pipelines/train_reclassifier_default.viame_csv.conf )
endif()

# Darknet specific configs
if( VIAME_ENABLE_DARKNET )
  set( CORE_PIPELINE_FILES ${CORE_PIPELINE_FILES}
    pipelines/train_detector_darknet_yolo_640.viame_csv.conf
    pipelines/train_detector_darknet_yolo_704.habcam.conf
    pipelines/train_detector_darknet_yolo_704.kw18.conf
    pipelines/train_detector_darknet_yolo_full.viame_csv.conf )
endif()

# Configs used exclusively for VIEW
if( VIAME_ENABLE_VIVIA )
  set( GUI_PIPELINE_FILES
    pipelines/embedded_single_stream/filter_debayer.pipe
    pipelines/embedded_single_stream/filter_debayer_and_enhance.pipe
    pipelines/embedded_single_stream/filter_enhance.pipe
    pipelines/embedded_single_stream/fish_detector_without_motion.pipe
    pipelines/embedded_single_stream/fish_tracker.pipe
    pipelines/embedded_single_stream/generate_empty_frame_lbls_0001fr.pipe
    pipelines/embedded_single_stream/generate_empty_frame_lbls_0010fr.pipe
    pipelines/embedded_single_stream/generate_empty_frame_lbls_0100fr.pipe
    pipelines/embedded_single_stream/generate_empty_frame_lbls_1000fr.pipe
    pipelines/embedded_single_stream/generic_proposal_detector.pipe
    pipelines/embedded_single_stream/generic_proposal_tracker.pipe
    pipelines/embedded_single_stream/local_deep_detector.pipe
    pipelines/embedded_single_stream/local_deep_detector_with_def_tracker.pipe
    pipelines/embedded_single_stream/local_deep_detector_with_stab_tracker.pipe
    pipelines/embedded_single_stream/local_svm_detector.pipe
    pipelines/embedded_single_stream/local_svm_detector_with_def_tracker.pipe
    pipelines/embedded_single_stream/local_svm_frame_classifier.pipe )

  if( VIAME_ENABLE_DARKNET )
    set( GUI_PIPELINE_FILES ${GUI_PIPELINE_FILES}
      pipelines/embedded_single_stream/train_deep_detector_yolo.pipe )
  endif()

  if( VIAME_ENABLE_PYTORCH )
    set( GUI_PIPELINE_FILES ${GUI_PIPELINE_FILES}
      pipelines/embedded_single_stream/train_deep_detector_cfrnn.pipe )
  endif()

  if( VIAME_ENABLE_TENSORFLOW AND VIAME_DOWNLOAD_MODELS-ARCTIC-SEAL )
    set( GUI_PIPELINE_FILES ${GUI_PIPELINE_FILES}
      pipelines/embedded_single_stream/arctic_seal_eo_tf_detector.pipe
      pipelines/embedded_single_stream/arctic_seal_ir_tf_detector.pipe )
  endif()

  if( VIAME_ENABLE_DARKNET AND VIAME_DOWNLOAD_MODELS-ARCTIC-SEAL )
    set( GUI_PIPELINE_FILES ${GUI_PIPELINE_FILES}
      pipelines/embedded_single_stream/arctic_seal_eo_tiny_yolo_detector.pipe
      pipelines/embedded_single_stream/arctic_seal_eo_tiny_yolo_tracker.pipe
      pipelines/embedded_single_stream/arctic_seal_eo_tiny_yolo_tracker_alt2.pipe
      pipelines/embedded_single_stream/arctic_seal_eo_tiny_yolo_tracker_alt3.pipe
      pipelines/embedded_single_stream/arctic_seal_eo_yolo_detector.pipe
      pipelines/embedded_single_stream/arctic_seal_ir_tiny_yolo_detector.pipe
      pipelines/embedded_single_stream/arctic_seal_ir_yolo_detector.pipe )
  endif()

  if( VIAME_ENABLE_PYTORCH AND VIAME_ENABLE_PYTORCH-PYSOT )
    set( GUI_PIPELINE_FILES ${GUI_PIPELINE_FILES}
      pipelines/embedded_single_stream/track_user_selections_fast.pipe )

    if( VIAME_ENABLE_PYTORCH-MDNET )
      set( GUI_PIPELINE_FILES ${GUI_PIPELINE_FILES}
        pipelines/embedded_single_stream/track_user_selections_fusion.pipe )
    endif()
  endif()

  install( FILES       ${GUI_PIPELINE_FILES}
           DESTINATION configs/pipelines/embedded_single_stream )
endif()

if( VIAME_ENABLE_BURNOUT )
  set( CORE_PIPELINE_FILES ${CORE_PIPELINE_FILES}
    pipelines/burnout_det_descriptor.cfe.conf
    pipelines/burnout_det_descriptors.conf
    pipelines/burnout_enhancement.conf
    pipelines/burnout_trk_descriptor.cfe.conf
    pipelines/burnout_trk_descriptors.conf
    pipelines/burnout_pixel_classification.conf
    pipelines/burnout_pixel_detection.conf
    pipelines/burnout_train_classifier.conf
    pipelines/detector_burnout.pipe
    pipelines/filter_burnout.pipe )
endif()

if( VIAME_ENABLE_SMQTK )
  set( CORE_PIPELINE_FILES ${CORE_PIPELINE_FILES}
    pipelines/smqtk_compute_hashes.json
    pipelines/smqtk_desc_index.json
    pipelines/smqtk_make_balltree.json
    pipelines/smqtk_nn_index.json
    pipelines/smqtk_postgres_ingest.json
    pipelines/smqtk_query.pipe
    pipelines/smqtk_train_itq.json
    pipelines/train_detector_svm_over_fish_detections.viame_csv.conf
    pipelines/train_detector_svm_over_generic_detections.viame_csv.conf
    pipelines/train_frame_classifier_svm_over_resnet.viame_csv.conf )
endif()

if( VIAME_ENABLE_OPENCV )
  set( CORE_PIPELINE_FILES ${CORE_PIPELINE_FILES}
    pipelines/register_using_homographies.pipe
    pipelines/register_multimodal_unsync_ocv.pipe )
endif()

if( VIAME_ENABLE_PYTORCH AND VIAME_ENABLE_PYTORCH-MMDET )
#  set( CORE_PIPELINE_FILES ${CORE_PIPELINE_FILES}
#    pipelines/train_detector_mmdet_cfrnn.habcam.conf
#    pipelines/train_detector_mmdet_cfrnn.viame_csv.conf)
endif()

if( VIAME_ENABLE_PYTORCH-NETHARN )
  set( CORE_PIPELINE_FILES ${CORE_PIPELINE_FILES}
    pipelines/train_detector_netharn_cfrnn.viame_csv.conf
    pipelines/train_detector_netharn_cfrnn.continue.viame_csv.conf
    pipelines/train_detector_netharn_cfrnn.habcam.conf
    pipelines/train_detector_netharn_cfrnn_motion.viame_csv.conf
    pipelines/train_detector_netharn_cfrnn_special.viame_csv.conf
    pipelines/train_detector_netharn_cfrnn_wtf.viame_csv.conf
    pipelines/train_detector_netharn_mask_rcnn.viame_csv.conf
    pipelines/train_frame_classifier_netharn_efficientnet.viame_csv.conf
    pipelines/train_frame_classifier_netharn_efficientnet.continue.viame_csv.conf
    pipelines/train_frame_classifier_netharn_resnet.viame_csv.conf
    pipelines/train_frame_classifier_netharn_resnet.continue.viame_csv.conf
    pipelines/train_reclassifier_netharn_resnet.viame_csv.conf )
endif()

if( VIAME_ENABLE_PYTORCH AND VIAME_ENABLE_PYTORCH-PYSOT )
  set( CORE_PIPELINE_FILES ${CORE_PIPELINE_FILES}
    pipelines/common_short_term_tracker.pipe
    pipelines/utility_track_selections_default_mask.pipe
    pipelines/utility_track_selections_fish_box_only.pipe )
endif()

if( VIAME_ENABLE_ITK )
  set( CORE_PIPELINE_FILES ${CORE_PIPELINE_FILES}
    pipelines/register_multimodal_sync_itk.pipe
    pipelines/register_multimodal_unsync_itk.pipe )
endif()

if( VIAME_ENABLE_TENSORFLOW AND VIAME_DOWNLOAD_MODELS-ARCTIC-SEAL )
  set( CORE_PIPELINE_FILES ${CORE_PIPELINE_FILES}
    pipelines/common_arctic_seal_eo_tf_detector.pipe
    pipelines/common_arctic_seal_ir_tf_detector.pipe
    pipelines/detector_arctic_seal_eo_tf.pipe
    pipelines/detector_arctic_seal_ir_tf.pipe )
endif()

if( VIAME_ENABLE_DARKNET AND VIAME_DOWNLOAD_MODELS-ARCTIC-SEAL )
  set( CORE_PIPELINE_FILES ${CORE_PIPELINE_FILES}
    pipelines/common_arctic_seal_eo_yolo_detector.pipe
    pipelines/common_arctic_seal_eo_tiny_yolo_detector.pipe
    pipelines/common_arctic_seal_ir_yolo_detector.pipe
    pipelines/common_arctic_seal_ir_tiny_yolo_detector.pipe
    pipelines/detector_arctic_seal_fusion_yolo.pipe
    pipelines/detector_arctic_seal_eo_yolo.pipe
    pipelines/detector_arctic_seal_ir_yolo.pipe )
endif()

if( VIAME_ENABLE_MATLAB )
  set( CORE_PIPELINE_FILES ${CORE_PIPELINE_FILES}
    pipelines/detector_matlab_camtrawl.pipe )
endif()

if( VIAME_ENABLE_CAFFE )
  set( CORE_PIPELINE_FILES ${CORE_PIPELINE_FILES}
    pipelines/tracker_fish.sfd.cfe.pipe
    pipelines/index_default.sfd.cfe.pipe )
endif()

if( VIAME_ENABLE_SEAL OR VIAME_DOWNLOAD_MODELS-ARCTIC-SEAL )
  set( DUAL_EMBEDDED_FILES
    pipelines/embedded_dual_stream/local_trained_eo_detector.pipe
    pipelines/embedded_dual_stream/local_trained_ir_detector.pipe )

  if( VIAME_ENABLE_PYTORCH-NETHARN )
    set( DUAL_EMBEDDED_FILES ${DUAL_EMBEDDED_FILES}
      pipelines/embedded_dual_stream/train_cfrnn_eo.pipe
      pipelines/embedded_dual_stream/train_cfrnn_ir.pipe )
  endif()

  if( VIAME_ENABLE_DARKNET )
    set( DUAL_EMBEDDED_FILES ${DUAL_EMBEDDED_FILES}
      pipelines/embedded_dual_stream/train_yolo_eo.pipe
      pipelines/embedded_dual_stream/train_yolo_ir.pipe )
  endif()

  if( VIAME_ENABLE_PYTORCH-PYSOT )
    set( DUAL_EMBEDDED_FILES ${DUAL_EMBEDDED_FILES}
      pipelines/embedded_dual_stream/track_user_selections_fast.pipe )
  endif()

  if( VIAME_DOWNLOAD_MODELS-ARCTIC-SEAL )
    set( DUAL_EMBEDDED_FILES ${DUAL_EMBEDDED_FILES}
      pipelines/embedded_dual_stream/arctic_seal_tf_eo_only.pipe
      pipelines/embedded_dual_stream/arctic_seal_tf_ir_only.pipe
      pipelines/embedded_dual_stream/arctic_seal_tf_ir_to_eo_frame_trigger.pipe
      pipelines/embedded_dual_stream/arctic_seal_tiny_yolo_eo_only.pipe
      pipelines/embedded_dual_stream/arctic_seal_tiny_yolo_eo_tracker.pipe
      pipelines/embedded_dual_stream/arctic_seal_tiny_yolo_eo_tracker_alt.pipe
      pipelines/embedded_dual_stream/arctic_seal_yolo_eo_ir_early_fusion.pipe
      pipelines/embedded_dual_stream/arctic_seal_yolo_eo_only.pipe
      pipelines/embedded_dual_stream/arctic_seal_yolo_ir_only.pipe
      pipelines/embedded_dual_stream/arctic_seal_yolo_ir_to_eo_frame_trigger.pipe
      pipelines/embedded_dual_stream/arctic_seal_yolo_ir_to_eo_region_trigger.pipe
      pipelines/embedded_dual_stream/arctic_seal_yolo_ir_to_tiny_eo_region_tigger.pipe )
  endif()

  install( FILES       ${DUAL_EMBEDDED_FILES}
           DESTINATION configs/pipelines/embedded_dual_stream )
endif()

install( FILES       ${CORE_PIPELINE_FILES}
         DESTINATION configs/pipelines )
install( DIRECTORY   pipelines/templates
         DESTINATION configs/pipelines )

if( VIAME_DOWNLOAD_MODELS )
  install( DIRECTORY   pipelines/models
           DESTINATION configs/pipelines )

  if( VIAME_DOWNLOAD_MODELS-ARCTIC-SEAL )
    install( DIRECTORY   pipelines/transformations
             DESTINATION configs/pipelines )
  endif()
endif()
