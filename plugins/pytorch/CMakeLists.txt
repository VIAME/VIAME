
kwiver_create_python_init( viame/arrows/pytorch )

kwiver_add_python_module(
  ${CMAKE_CURRENT_SOURCE_DIR}/train_mmdet_detector.py
  viame/arrows/pytorch
  train_mmdet_detector )
