
kwiver_create_python_init( arrows/pytorch )

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

# Common Utilities
kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/utilities.py
  arrows/pytorch
  utilities )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/kwcoco_train_detector.py
  arrows/pytorch
  kwcoco_train_detector )

if( VIAME_ENABLE_PYTORCH-NETHARN )

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
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    processes/pytorch
    __init__ )

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

if( VIAME_ENABLE_LEARN )

  add_subdirectory( cutler )
  add_subdirectory( remax )

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
