
kwiver_create_python_init( pytorch )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/utilities.py
  pytorch
  utilities )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/kwcoco_train_detector.py
  pytorch
  kwcoco_train_detector )

if( VIAME_ENABLE_PYTORCH-LEARN )
  add_subdirectory( learn )
endif()

add_subdirectory( srnn )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/srnn_tracker.py
  pytorch
  srnn_tracker )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/srnn_trainer.py
  pytorch
  srnn_trainer )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/deepsort_tracker.py
  pytorch
  deepsort_tracker )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/deepsort_trainer.py
  pytorch
  deepsort_trainer )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/botsort_tracker.py
  pytorch
  botsort_tracker )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/botsort_trainer.py
  pytorch
  botsort_trainer )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  pytorch
  __init__ )

if( VIAME_ENABLE_PYTORCH-MMDET )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mmdet_compatibility.py
    pytorch
    mmdet_compatibility )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mmdet_detector.py
    pytorch
    mmdet_detector )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mmdet_trainer.py
    pytorch
    mmdet_trainer )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mmdet_launcher.py
    pytorch
    mmdet_launcher )

endif()

if( VIAME_ENABLE_PYTORCH-NETHARN )

  add_subdirectory( netharn )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/netharn_classifier.py
    pytorch
    netharn_classifier )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/netharn_detector.py
    pytorch
    netharn_detector )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/netharn_trainer.py
    pytorch
    netharn_trainer )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/netharn_refiner.py
    pytorch
    netharn_refiner )

endif()

if( VIAME_ENABLE_PYTORCH-HUGGINGFACE )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/huggingface_zeroshot_detector.py
    pytorch
    huggingface_zeroshot_detector )

endif()

if( VIAME_ENABLE_PYTORCH-MIT-YOLO )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mit_yolo_trainer.py
    pytorch
    mit_yolo_trainer )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mit_yolo_detector.py
    pytorch
    mit_yolo_detector )

endif()

if( VIAME_ENABLE_PYTORCH-ULTRALYTICS )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/ultralytics_detector.py
    pytorch
    ultralytics_detector )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/ultralytics_trainer.py
    pytorch
    ultralytics_trainer )

endif()

if( VIAME_ENABLE_ONNX AND ( VIAME_ENABLE_DARKNET OR VIAME_ENABLE_PYTORCH-MMDET ) )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/convert_to_onnx_process.py
    pytorch
    convert_to_onnx_process )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/onnx/crcnn2onnx.py
    pytorch
    crcnn2onnx )

endif()

if( VIAME_ENABLE_PYTORCH-MDNET )

  add_subdirectory( mdnet )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mdnet_tracker.py
    pytorch
    mdnet_tracker )

endif()

if( VIAME_ENABLE_PYTORCH-SIAMMASK )

  add_subdirectory( siammask )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/siammask_tracker.py
    pytorch
    siammask_tracker )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/siammask_trainer.py
    pytorch
    siammask_trainer )

endif()

if( VIAME_ENABLE_PYTORCH-LEARN )

  add_subdirectory( remax )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/maskcut_detector.py
    pytorch
    maskcut_detector )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/cutler_trainer.py
    pytorch
    cutler_trainer )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/remax_convnext_detector.py
    pytorch
    remax_convnext_detector )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/remax_convnext_trainer.py
    pytorch
    remax_convnext_trainer )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/remax_dino_detector.py
    pytorch
    remax_dino_detector )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/remax_dino_trainer.py
    pytorch
    remax_dino_trainer )

endif()

if( VIAME_ENABLE_PYTORCH-SAM2 )

    kwiver_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/sam2_refiner.py
      pytorch
      sam2_refiner )

    kwiver_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/sam2_interactive.py
      pytorch
      sam2_interactive )

endif()

if( VIAME_ENABLE_PYTORCH-SAM3 )

    kwiver_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/sam3_interactive.py
      pytorch
      sam3_interactive )

    kwiver_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/sam3_tracker.py
      pytorch
      sam3_tracker )

endif()

if( VIAME_ENABLE_PYTORCH-STEREO )

    kwiver_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/foundation_stereo.py
      pytorch
      foundation_stereo )

endif()

if( VIAME_ENABLE_PYTORCH-RF-DETR )

    kwiver_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/rf_detr_detector.py
      pytorch
      rf_detr_detector )

    kwiver_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/rf_detr_trainer.py
      pytorch
      rf_detr_trainer )

endif()

if( VIAME_ENABLE_PYTORCH-LITDET )

    kwiver_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/litdet_detector.py
      pytorch
      litdet_detector )

    kwiver_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/litdet_trainer.py
      pytorch
      litdet_trainer )

endif()

if( VIAME_ENABLE_PYTORCH-DETECTRON2 )

    kwiver_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/detectron2_trainer.py
      pytorch
      detectron2_trainer )

endif()

if( VIAME_ENABLE_PYTORCH-VISION )
  add_subdirectory( torchvision )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/torchvision_descriptors.py
    pytorch
    torchvision_descriptors )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/torchvision_augment_process.py
    pytorch
    torchvision_augment_process )
endif()
