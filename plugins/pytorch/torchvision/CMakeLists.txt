###
# Torchvision-based feature extractors

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  sprokit/processes/pytorch/torchvision
  __init__ )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/resnet_feature_extractor.py
  sprokit/processes/pytorch/torchvision
  resnet_feature_extractor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/alexnet_feature_extractor.py
  sprokit/processes/pytorch/torchvision
  alexnet_feature_extractor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/enet_feature_extractor.py
  sprokit/processes/pytorch/torchvision
  enet_feature_extractor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/resnet_augmenter.py
  sprokit/processes/pytorch/torchvision
  resnet_augmenter )
