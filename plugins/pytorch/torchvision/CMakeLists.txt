###
# Torchvision-based feature extractors

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  pytorch/torchvision
  __init__ )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/resnet_feature_extractor.py
  pytorch/torchvision
  resnet_feature_extractor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/alexnet_feature_extractor.py
  pytorch/torchvision
  alexnet_feature_extractor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/enet_feature_extractor.py
  pytorch/torchvision
  enet_feature_extractor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/resnet_augmenter.py
  pytorch/torchvision
  resnet_augmenter )
