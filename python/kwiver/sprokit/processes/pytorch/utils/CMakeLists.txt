###
# Loadable algorithms and processes

kwiver_create_python_init( kwiver/processes/pytorch/utils )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/siamese_feature_extractor.py
  kwiver/processes/pytorch/utils
  siamese_feature_extractor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/resnet_feature_extractor.py
  kwiver/processes/pytorch/utils
  resnet_feature_extractor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/augmented_resnet_feature_extractor.py
  kwiver/processes/pytorch/utils
  augmented_resnet_feature_extractor )
  
kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/alexnet_feature_extractor.py
  kwiver/processes/pytorch/utils
  alexnet_feature_extractor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/grid.py
  kwiver/processes/pytorch/utils
  grid )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/track.py
  kwiver/processes/pytorch/utils
  track )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/models.py
  kwiver/processes/pytorch/utils
  models )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/srnn_matching.py
  kwiver/processes/pytorch/utils
  srnn_matching )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/gt_bbox.py
  kwiver/processes/pytorch/utils
  gt_bbox )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/iou_tracker.py
  kwiver/processes/pytorch/utils
  iou_tracker )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/parse_gpu_list.py
  kwiver/processes/pytorch/utils
  parse_gpu_list )
