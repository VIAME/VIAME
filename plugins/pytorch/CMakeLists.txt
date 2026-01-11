
kwiver_create_python_init( arrows/pytorch )
kwiver_create_python_init( pytorch )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/utilities.py
  pytorch
  utilities )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/kwcoco_train_detector.py
  arrows/pytorch
  kwcoco_train_detector )

if( VIAME_ENABLE_PYTORCH-LEARN )
  add_subdirectory( learn )
endif()

add_subdirectory( srnn )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/srnn_tracker.py
  processes/pytorch
  srnn_tracker )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  processes/pytorch
  __init__ )

if( VIAME_ENABLE_PYTORCH-MMDET )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mmdet_compatibility.py
    arrows/pytorch
    mmdet_compatibility )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mmdet_detector.py
    arrows/pytorch
    mmdet_detector )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mmdet_trainer.py
    arrows/pytorch
    mmdet_trainer )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mmdet_launcher.py
    arrows/pytorch
    mmdet_launcher )

endif()

if( VIAME_ENABLE_PYTORCH-NETHARN )

  add_subdirectory( netharn )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/netharn_classifier.py
    arrows/pytorch
    netharn_classifier )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/netharn_detector.py
    arrows/pytorch
    netharn_detector )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/netharn_trainer.py
    arrows/pytorch
    netharn_trainer )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/netharn_refiner.py
    arrows/pytorch
    netharn_refiner )

endif()

if( VIAME_ENABLE_PYTORCH-HUGGINGFACE )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/huggingface_zeroshot_detector.py
    arrows/pytorch
    huggingface_zeroshot_detector )

endif()

if( VIAME_ENABLE_PYTORCH-MIT-YOLO )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mit_yolo_trainer.py
    arrows/pytorch
    mit_yolo_trainer )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mit_yolo_detector.py
    arrows/pytorch
    mit_yolo_detector )

endif()

if( VIAME_ENABLE_PYTORCH-ULTRALYTICS )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/ultralytics_detector.py
    arrows/pytorch
    ultralytics_detector )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/ultralytics_trainer.py
    arrows/pytorch
    ultralytics_trainer )

endif()

if( VIAME_ENABLE_ONNX AND ( VIAME_ENABLE_DARKNET OR VIAME_ENABLE_PYTORCH-MMDET ) )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/convert_to_onnx_process.py
    processes/pytorch
    convert_to_onnx_process )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/onnx_conversion_tools/crcnn2onnx.py
    arrows/pytorch
    crcnn2onnx )

endif()

if( VIAME_ENABLE_PYTORCH-MDNET )

  add_subdirectory( mdnet )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mdnet_tracker.py
    arrows/pytorch
    mdnet_tracker )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mdnet_tracker_process.py
    processes/pytorch
    mdnet_tracker_process )

endif()

if( VIAME_ENABLE_PYTORCH-SIAMMASK )

  add_subdirectory( siammask )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/siammask_tracker.py
    processes/pytorch
    siammask_tracker )

endif()

if( VIAME_ENABLE_PYTORCH-LEARN )

  add_subdirectory( remax )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/maskcut_detector.py
    arrows/pytorch
    maskcut_detector )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/cutler_trainer.py
    arrows/pytorch
    cutler_trainer )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/remax_convnext_detector.py
    arrows/pytorch
    remax_convnext_detector )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/remax_convnext_trainer.py
    arrows/pytorch
    remax_convnext_trainer )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/remax_dino_detector.py
    arrows/pytorch
    remax_dino_detector )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/remax_dino_trainer.py
    arrows/pytorch
    remax_dino_trainer )

endif()

if( VIAME_ENABLE_PYTORCH-SAM )

    kwiver_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/sam2_refiner.py
      arrows/pytorch
      sam2_refiner )

endif()

if( VIAME_ENABLE_PYTORCH-STEREO )

    kwiver_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/foundation_stereo.py
      arrows/pytorch
      foundation_stereo )

endif()

if( VIAME_ENABLE_PYTORCH-RF-DETR )

    kwiver_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/rf_detr_detector.py
      arrows/pytorch
      rf_detr_detector )

    kwiver_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/rf_detr_trainer.py
      arrows/pytorch
      rf_detr_trainer )

endif()

if( VIAME_ENABLE_PYTORCH-VISION )
  add_subdirectory( torchvision )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/torchvision_descriptors.py
    arrows/pytorch
    torchvision_descriptors )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/desc_augmentation_process.py
    processes/pytorch
    desc_augmentation_process )
endif()
