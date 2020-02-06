###
# Loadable algorithms and processes

kwiver_create_python_init( kwiver/arrows/pytorch )

if( KWIVER_ENABLE_PROCESSES )
  kwiver_create_python_init( kwiver/processes/pytorch )

  kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/srnn_tracker.py
    kwiver/processes/pytorch
    srnn_tracker )

  kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/resnet_descriptors.py
    kwiver/processes/pytorch
    resnet_descriptors )

  kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/resnet_augmentation.py
    kwiver/processes/pytorch
    resnet_augmentation )

  kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/alexnet_descriptors.py
    kwiver/processes/pytorch
    alexnet_descriptors )

  kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/seg_detection.py
    kwiver/processes/pytorch
    seg_detection )

  if( KWIVER_ENABLE_PYTORCH_PYSOT )
    kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/pysot_tracker.py
      kwiver/processes/pytorch
      pysot_tracker )
  endif()
endif()

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/siamese_feature_extractor.py
  kwiver/arrows/pytorch
  siamese_feature_extractor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/resnet_feature_extractor.py
  kwiver/arrows/pytorch
  resnet_feature_extractor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/augmented_resnet_feature_extractor.py
  kwiver/arrows/pytorch
  augmented_resnet_feature_extractor )
  
kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/alexnet_feature_extractor.py
  kwiver/arrows/pytorch
  alexnet_feature_extractor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/grid.py
  kwiver/arrows/pytorch
  grid )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/track.py
  kwiver/arrows/pytorch
  track )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/models.py
  kwiver/arrows/pytorch
  models )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/srnn_matching.py
  kwiver/arrows/pytorch
  srnn_matching )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/gt_bbox.py
  kwiver/arrows/pytorch
  gt_bbox )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/seg_utils.py
  kwiver/arrows/pytorch
  seg_utils )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/fcn_segmentation.py
  kwiver/arrows/pytorch
  fcn_segmentation )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/fcn_models.py
  kwiver/arrows/pytorch
  fcn_models )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/iou_tracker.py
  kwiver/arrows/pytorch
  iou_tracker )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/parse_gpu_list.py
  kwiver/arrows/pytorch
  parse_gpu_list )
