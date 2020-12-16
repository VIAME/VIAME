###
# Loadable algorithms and processes

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  sprokit/processes/pytorch/utils
  __init__.py )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/siamese_feature_extractor.py
  sprokit/processes/pytorch/utils
  siamese_feature_extractor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/resnet_feature_extractor.py
  sprokit/processes/pytorch/utils
  resnet_feature_extractor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/augmented_resnet_feature_extractor.py
  sprokit/processes/pytorch/utils
  augmented_resnet_feature_extractor )
  
kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/alexnet_feature_extractor.py
  sprokit/processes/pytorch/utils
  alexnet_feature_extractor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/grid.py
  sprokit/processes/pytorch/utils
  grid )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/track.py
  sprokit/processes/pytorch/utils
  track )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/models.py
  sprokit/processes/pytorch/utils
  models )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/srnn_matching.py
  sprokit/processes/pytorch/utils
  srnn_matching )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/gt_bbox.py
  sprokit/processes/pytorch/utils
  gt_bbox )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/iou_tracker.py
  sprokit/processes/pytorch/utils
  iou_tracker )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/parse_gpu_list.py
  sprokit/processes/pytorch/utils
  parse_gpu_list )
