# What is left of `plugins/pytorch`: the trackers and trainers (P2-T06,
# P2-T07), the stereo and descriptor modules (P2-T06), and the vendored
# netharn, srnn, siammask, mdnet and learn trees. The detectors, classifiers
# and segmenters are gated subpackages of `library/object_detectors`,
# `library/classifiers` and `library/segmentation` since P2-T05, and the
# shared helpers that were `utilities.py` are `viame.object_detectors.base`.


viame_create_python_init( pytorch )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/kwcoco_train_detector.py
  pytorch
  kwcoco_train_detector )

if( VIAME_ENABLE_PYTORCH-LEARN )
  add_subdirectory( learn )
endif()

add_subdirectory( minima_loftr )

add_subdirectory( srnn )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/srnn_tracker.py
  pytorch
  srnn_tracker )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/srnn_trainer.py
  pytorch
  srnn_trainer )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/deepsort_tracker.py
  pytorch
  deepsort_tracker )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/deepsort_trainer.py
  pytorch
  deepsort_trainer )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/botsort_tracker.py
  pytorch
  botsort_tracker )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/botsort_trainer.py
  pytorch
  botsort_trainer )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/motr_tracker.py
  pytorch
  motr_tracker )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/motr_trainer.py
  pytorch
  motr_trainer )

viame_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  pytorch
  __init__ )

if( VIAME_ENABLE_ONNX)

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/convert_to_onnx_process.py
    pytorch
    convert_to_onnx_process )

  if( VIAME_ENABLE_DARKNET )
    viame_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/onnx_exporters/darknet_to_onnx.py
      pytorch
      darknet_to_onnx )
  endif()

endif()

if( VIAME_ENABLE_PYTORCH-MMDET )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mmdet_trainer.py
    pytorch
    mmdet_trainer )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mmdet_launcher.py
    pytorch
    mmdet_launcher )

endif()

if( VIAME_ENABLE_PYTORCH-NETHARN )

  add_subdirectory( netharn )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/netharn_trainer.py
    pytorch
    netharn_trainer )

  
    if( VIAME_ENABLE_ONNX)

      viame_add_python_module(
        ${CMAKE_CURRENT_SOURCE_DIR}/onnx_exporters/netharn_mmdet_to_onnx.py
        pytorch
        netharn_mmdet_to_onnx )

      viame_add_python_module(
        ${CMAKE_CURRENT_SOURCE_DIR}/onnx_exporters/netharn_clf_to_onnx.py
        pytorch
        netharn_clf_to_onnx )

    endif()

endif()

if( VIAME_ENABLE_PYTORCH-MIT-YOLO )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mit_yolo_trainer.py
    pytorch
    mit_yolo_trainer )

  if( VIAME_ENABLE_ONNX)

      viame_add_python_module(
        ${CMAKE_CURRENT_SOURCE_DIR}/onnx_exporters/yolomit_to_onnx.py
        pytorch
        yolomit_to_onnx )

    endif()

endif()

if( VIAME_ENABLE_PYTORCH-ULTRALYTICS )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/ultralytics_trainer.py
    pytorch
    ultralytics_trainer )

endif()

if( VIAME_ENABLE_PYTORCH-MDNET )

  add_subdirectory( mdnet )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mdnet_tracker.py
    pytorch
    mdnet_tracker )

endif()

if( VIAME_ENABLE_PYTORCH-SIAMMASK )

  add_subdirectory( siammask )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/siammask_tracker.py
    pytorch
    siammask_tracker )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/siammask_trainer.py
    pytorch
    siammask_trainer )

endif()

if( VIAME_ENABLE_PYTORCH-LEARN )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/cutler_trainer.py
    pytorch
    cutler_trainer )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/remax_convnext_trainer.py
    pytorch
    remax_convnext_trainer )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/remax_dino_trainer.py
    pytorch
    remax_dino_trainer )

endif()

if( VIAME_ENABLE_PYTORCH-SAM3 )

    viame_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/sam3_tracker.py
      pytorch
      sam3_tracker )

    viame_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/sam3_trainer.py
      pytorch
      sam3_trainer )

endif()

if( VIAME_ENABLE_PYTORCH-STEREO )

    viame_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/foundation_stereo.py
      pytorch
      foundation_stereo )

    viame_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/fast_foundation_stereo.py
      pytorch
      fast_foundation_stereo )

endif()

if( VIAME_ENABLE_PYTORCH-DINO3 )

    viame_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/dino_matcher.py
      pytorch
      dino_matcher )

endif()

if( VIAME_ENABLE_PYTORCH-RF-DETR )

    viame_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/rf_detr_trainer.py
      pytorch
      rf_detr_trainer )

    viame_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/rf_detr_launcher.py
      pytorch
      rf_detr_launcher )

endif()

if( VIAME_ENABLE_PYTORCH-LITDET )

    viame_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/litdet_trainer.py
      pytorch
      litdet_trainer )

endif()

if( VIAME_ENABLE_PYTORCH-DETECTRON2 )

    viame_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/detectron2_trainer.py
      pytorch
      detectron2_trainer )

endif()

if( VIAME_ENABLE_PYTORCH-VISION )
  add_subdirectory( torchvision )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/torchvision_descriptors.py
    pytorch
    torchvision_descriptors )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/torchvision_augment_process.py
    pytorch
    torchvision_augment_process )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/pair_stereo_tracks.py
    pytorch
    pair_stereo_tracks )
endif()

if( VIAME_ENABLE_PYTORCH-SLEAP )
  foreach( MODULE sleap_common sleap_trainer sleap_launcher )
    viame_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/${MODULE}.py
      pytorch
      ${MODULE} )
  endforeach()
endif()
