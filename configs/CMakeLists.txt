
###
# Download and install enabled model files
##
if( VIAME_DOWNLOAD_MODELS )

  # GUI Context View
  DownloadAndExtract(
    https://data.kitware.com/api/v1/item/5afe6c018d777f15ebe1d6f2/download
    7e8800cb11a1a491140e7f58e9053030
    ${VIAME_DOWNLOAD_DIR}/models-blue_marble_context-v1.0.0.tar.gz
    ${CMAKE_CURRENT_SOURCE_DIR}/gui-params )

  # PyTorch Tracking Models
  if( VIAME_ENABLE_PYTORCH AND VIAME_DOWNLOAD_MODELS-PYTORCH )
    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5b6890a38d777f06857c1f22/download
      51b979d61df667b4751000192426f106
      ${VIAME_DOWNLOAD_DIR}/models-pytorch_tracking-v1.1.0.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )

    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5bad56178d777f06b908c47c/download
      3fe6c43e8bf6ab49cb8f6687f38bdbd6
      ${VIAME_DOWNLOAD_DIR}/models-pytorch_descriptors-v1.1.0.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
  endif()

  # Default Caffe Models
  if( VIAME_ENABLE_CAFFE )
    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5afe6e1e8d777f15ebe1d6fc/download
      2cdbe7ca51b49b12e59911aadec20e98
      ${VIAME_DOWNLOAD_DIR}/models-default_caffe_descriptor-v1.0.0.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
  endif()

  # Default YOLO Configs
  if( VIAME_ENABLE_YOLO )
    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5b5f00d18d777f06857c17fa/download
      fad64978df6322605530a5cc6274467c
      ${VIAME_DOWNLOAD_DIR}/models-default_yolo_v2-v2.1.0.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )

    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5c423a5f8d777f072b0ba58f/download
      d568eab97562b39500d774bc072726c0
      ${VIAME_DOWNLOAD_DIR}/models-yolo_v2_seed_model-v1.2.0.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )

    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5c423a608d777f072b0ba599/download
      1142628e15d6dbe70157a925bbe0e1f2
      ${VIAME_DOWNLOAD_DIR}/models-yolo_v3_seed_model-v1.2.0.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )

    if( VIAME_DOWNLOAD_MODELS-HABCAM )
      DownloadAndExtract(
        https://data.kitware.com/api/v1/item/5b5e27cc8d777f06857c17c2/download
        0ee03ac4a199cc6f884b2f9702e59eb1
        ${VIAME_DOWNLOAD_DIR}/models-habcam-v2.0.0.tar.gz
        ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
    endif()

    if( VIAME_DOWNLOAD_MODELS-MOUSS )
      DownloadAndExtract(
        https://data.kitware.com/api/v1/item/5bad56158d777f06b908c473/download
        00aa39ecdc321b205ed54766441c65aa
        ${VIAME_DOWNLOAD_DIR}/models-mouss-v2.1.0.tar.gz
        ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
    endif()
  endif()

  # Default Faster-RCNN Models
  if( VIAME_ENABLE_FASTER_RCNN )
    DownloadAndExtract(
      https://data.kitware.com/api/v1/item/5afe74fc8d777f15ebe1d70c/download
      e4792d72721465a206580522c37d0000
      ${VIAME_DOWNLOAD_DIR}/models-default_faster_rcnn-v1.0.0.tar.gz
      ${CMAKE_CURRENT_SOURCE_DIR}/pipelines )
  endif()
endif()

###
# All subdirectories for this folder installed as-is
##
install( DIRECTORY gui-params
         DESTINATION configs )
install( DIRECTORY prj-linux
         DESTINATION configs )
install( DIRECTORY prj-windows
         DESTINATION configs )

###
# All files manually-included based on options
##
set( RUN_SCRIPT_FILES
  database_tool.py
  extract_video_frames.py
  process_video.py
  launch_search_interface.py
  launch_timeline_interface.py
  launch_annotation_interface.py
  generate_detection_plots.py
  score_results.py )

set( CORE_PIPELINE_FILES
  pipelines/common_default_descriptor.pipe
  pipelines/common_default_detector.pipe
  pipelines/common_default_detector_with_filter.pipe
  pipelines/common_default_initializer.pipe
  pipelines/common_default_input.pipe
  pipelines/common_default_input_with_downsampler.pipe
  pipelines/common_default_tracker.pipe
  pipelines/common_generic_detector.pipe
  pipelines/common_generic_detector_with_filter.pipe
  pipelines/database_apply_svm_models.pipe
  pipelines/detector_default.pipe
  pipelines/detector_extract_chips.pipe
  pipelines/detector_generic.pipe
  pipelines/detector_simple_hough.pipe
  pipelines/detector_svm_models.pipe
  pipelines/display_annotations.pipe
  pipelines/draw_detections_on_images.pipe
  pipelines/extract_chips_from_detections.pipe
  pipelines/filter_debayer_and_enhance.pipe
  pipelines/filter_enhance.pipe
  pipelines/filter_to_kwa.pipe
  pipelines/full_frame_classifier_svm.pipe
  pipelines/index_default.pipe
  pipelines/index_default.trk.pipe
  pipelines/index_existing.pipe
  pipelines/index_full_frame.pipe
  pipelines/index_generic.pipe
  pipelines/index_generic.trk.pipe
  pipelines/measurement_default.trk.pipe
  pipelines/measurement_depth_map.pipe
  pipelines/measurement.gmm_only.pipe
  pipelines/query_augment_image.pipe
  pipelines/query_image_exemplar.cfe.pipe
  pipelines/query_image_exemplar.pipe
  pipelines/query_retrieval_and_iqr.cfe.pipe
  pipelines/query_retrieval_and_iqr.pipe
  pipelines/query_video_exemplar.pipe
  pipelines/sql_init_table.sql
  pipelines/tracker_default.pipe
  pipelines/tracker_default.sfd.pipe
  pipelines/tracker_generic.pipe
  pipelines/tracker_svm_models.pipe
  pipelines/train_color_freq_aug.pipe
  pipelines/train_motion_and_color_freq_aug.pipe
  pipelines/train_motion_aug.pipe
  pipelines/train_split_and_stereo_aug.pipe
  pipelines/train_split_aug.pipe
  pipelines/train_warp_ir_to_eo_aug.pipe
  pipelines/train_yolo_544.viame_csv.conf
  pipelines/train_yolo_704.habcam.conf
  pipelines/train_yolo_704.kw18.conf
  pipelines/train_yolo_704.viame_csv.conf
  pipelines/train_yolo_960.habcam.conf
  pipelines/train_yolo_960.viame_csv.conf
  pipelines/train_yolo_wtf_704.viame_csv.conf )

if( VIAME_ENABLE_BURNOUT )
  set( CORE_PIPELINE_FILES ${CORE_PIPELINE_FILES}
    pipelines/burnout_det_descriptor.cfe.conf
    pipelines/burnout_det_descriptors.conf
    pipelines/burnout_enhancement.conf
    pipelines/burnout_trk_descriptor.cfe.conf
    pipelines/burnout_trk_descriptors.conf )
endif()

if( VIAME_ENABLE_SMQTK )
  set( CORE_PIPELINE_FILES ${CORE_PIPELINE_FILES}
    pipelines/smqtk_compute_hashes.json
    pipelines/smqtk_desc_index.json
    pipelines/smqtk_make_balltree.json
    pipelines/smqtk_nn_index.json
    pipelines/smqtk_postgres_ingest.json
    pipelines/smqtk_query.pipe
    pipelines/smqtk_train_itq.json )
endif()

if( VIAME_ENABLE_SMQTK )
  set( CORE_PIPELINE_FILES ${CORE_PIPELINE_FILES}
    pipelines/register_optical_and_thermal.pipe
    pipelines/register_optical_and_thermal_unsync.pipe )
endif()

if( VIAME_DOWNLOAD_MODELS-MOUSS )
  set( CORE_PIPELINE_FILES ${CORE_PIPELINE_FILES}
    pipelines/common_mouss_detector.pipe
    pipelines/common_mouss_detector_with_filter.pipe
    pipelines/index_mouss.pipe
    pipelines/index_mouss.no_desc.pipe
    pipelines/index_mouss.trk.no_desc.pipe
    pipelines/index_mouss.trk.pipe
    pipelines/tracker_mouss.pipe )
endif()

if( VIAME_DOWNLOAD_MODELS-HABCAM )
  set( CORE_PIPELINE_FILES ${CORE_PIPELINE_FILES}
    pipelines/detector_habcam_system.pipe )
endif()

if( VIAME_ENABLE_MATLAB )
  set( CORE_PIPELINE_FILES ${CORE_PIPELINE_FILES}
    pipelines/detector_lanl_scallop_finder.pipe
    pipelines/detector_matlab_camtrawl.pipe )
endif()

if( VIAME_ENABLE_SCALLOP_TK )
  set( CORE_PIPELINE_FILES ${CORE_PIPELINE_FILES}
    pipelines/detector_scallop_tk.pipe )
endif()

if( VIAME_ENABLE_CAFFE )
  set( CORE_PIPELINE_FILES ${CORE_PIPELINE_FILES}
    pipelines/tracker_default.sfd.cfe.pipe
    pipelines/index_default.sfd.cfe.pipe )
endif()

install( FILES       ${RUN_SCRIPT_FILES}
         DESTINATION configs )
install( FILES       ${CORE_PIPELINE_FILES}
         DESTINATION configs/pipelines )
install( DIRECTORY   pipelines/models
         DESTINATION configs/pipelines )


if( VIAME_ENABLE_VIVIA )
  set( GUI_PIPELINE_FILES
    pipelines/gui_embedded/default_detector.pipe
    pipelines/gui_embedded/default_tracker.pipe
    pipelines/gui_embedded/generic_detector.pipe
    pipelines/gui_embedded/generic_tracker.pipe
    pipelines/gui_embedded/local_deep_detector.pipe
    pipelines/gui_embedded/local_deep_tracker.pipe
    pipelines/gui_embedded/local_svm_detector.pipe
    pipelines/gui_embedded/local_svm_tracker.pipe
    pipelines/gui_embedded/train_deep_detector.pipe )

  if( VIAME_DOWNLOAD_MODELS-MOUSS )
    set( GUI_PIPELINE_FILES ${GUI_PIPELINE_FILES}
      pipelines/gui_embedded/mouss_detector.pipe
      pipelines/gui_embedded/mouss_tracker.pipe )
  endif()

  if( VIAME_DOWNLOAD_MODELS-HABCAM )
    set( GUI_PIPELINE_FILES ${GUI_PIPELINE_FILES}
      pipelines/gui_embedded/habcam_detector.pipe )
  endif()

  install( FILES       ${GUI_PIPELINE_FILES}
           DESTINATION configs/pipelines/gui_embedded )
endif()
