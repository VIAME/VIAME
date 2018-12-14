###
# Loadable algorithms

if( KWIVER_ENABLE_PYTORCH )

  kwiver_create_python_plugin_init(kwiver/arrows/pytorch)
  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/resnet_descriptors.py
    kwiver/processes
    resnet_descriptors)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/SRNN_tracking.py
    kwiver/processes
    SRNN_tracking)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/seg_detection.py
    kwiver/processes
    seg_detection)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/pytorch_augmentation.py
    kwiver/processes
    pytorch_augmentation)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/pytorch_siamese_f_extractor.py
    kwiver/arrows/pytorch
    pytorch_siamese_f_extractor)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/pytorch_resnet_f_extractor.py
    kwiver/arrows/pytorch
    pytorch_resnet_f_extractor)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/pytorch_aug_resnet_extractor.py
    kwiver/arrows/pytorch
    pytorch_aug_resnet_extractor)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/grid.py
    kwiver/arrows/pytorch
    grid)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/track.py
    kwiver/arrows/pytorch
    track)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/models.py
    kwiver/arrows/pytorch
    models)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/SRNN_matching.py
    kwiver/arrows/pytorch
    SRNN_matching)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/MOT_bbox.py
    kwiver/arrows/pytorch
    MOT_bbox)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/seg_utils.py
    kwiver/arrows/pytorch
    seg_utils)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/fcn_segmentation.py
    kwiver/arrows/pytorch
    fcn_segmentation)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/fcn_models.py
    kwiver/arrows/pytorch
    fcn_models)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/iou_tracking.py
    kwiver/arrows/pytorch
    iou_tracking)

  kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/parse_gpu_list.py
    kwiver/arrows/pytorch
    parse_gpu_list)
endif()
