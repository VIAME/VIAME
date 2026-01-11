###
# Torchvision-based feature extractors

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  arrows/pytorch/torchvision
  __init__ )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/resnet_feature_extractor.py
  arrows/pytorch/torchvision
  resnet_feature_extractor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/alexnet_feature_extractor.py
  arrows/pytorch/torchvision
  alexnet_feature_extractor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/enet_feature_extractor.py
  arrows/pytorch/torchvision
  enet_feature_extractor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/resnet_augmenter.py
  arrows/pytorch/torchvision
  resnet_augmenter )
