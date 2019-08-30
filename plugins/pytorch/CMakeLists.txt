
if( VIAME_ENABLE_PYTORCH-MMDET )

  kwiver_create_python_init( viame/arrows/pytorch )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mmdet_detector.py
    viame/arrows/pytorch
    mmdet_detector )

  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/mmdet_trainer.py
    viame/arrows/pytorch
    mmdet_trainer )

endif()
