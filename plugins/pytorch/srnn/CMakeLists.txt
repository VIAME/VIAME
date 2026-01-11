###
# SRNN tracker utilities

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  pytorch/srnn
  __init__ )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/track.py
  pytorch/srnn
  track )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/models.py
  pytorch/srnn
  models )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/srnn_matching.py
  pytorch/srnn
  srnn_matching )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/siamese_feature_extractor.py
  pytorch/srnn
  siamese_feature_extractor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/iou_tracker.py
  pytorch/srnn
  iou_tracker )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/gt_bbox.py
  pytorch/srnn
  gt_bbox )
