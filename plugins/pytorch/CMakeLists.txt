
if( VIAME_ENABLE_PYTORCH-MMDET )

  kwiver_create_python_init( viame/arrows/pytorch )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mmdet_compatibility.py
    viame/arrows/pytorch
    mmdet_compatibility )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mmdet_detector.py
    viame/arrows/pytorch
    mmdet_detector )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mmdet_trainer.py
    viame/arrows/pytorch
    mmdet_trainer )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mmdet_launcher.py
    viame/arrows/pytorch
    mmdet_launcher )

endif()


if( VIAME_ENABLE_PYTORCH-NETHARN )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/netharn_detector.py
    viame/arrows/pytorch
    netharn_detector )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/_netharn_detect_predict.py
    viame/arrows/pytorch
    _netharn_detect_predict )

endif()
