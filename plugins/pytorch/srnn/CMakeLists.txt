###
# SRNN tracker utilities

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  sprokit/processes/pytorch/srnn
  __init__ )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/track.py
  sprokit/processes/pytorch/srnn
  track )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/models.py
  sprokit/processes/pytorch/srnn
  models )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/srnn_matching.py
  sprokit/processes/pytorch/srnn
  srnn_matching )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/siamese_feature_extractor.py
  sprokit/processes/pytorch/srnn
  siamese_feature_extractor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/iou_tracker.py
  sprokit/processes/pytorch/srnn
  iou_tracker )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/gt_bbox.py
  sprokit/processes/pytorch/srnn
  gt_bbox )
