###
# Loadable algorithms and processes

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  sprokit/processes/pytorch
  __init__.py )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/srnn_tracker.py
  sprokit/processes/pytorch
  srnn_tracker )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/resnet_descriptors.py
  sprokit/processes/pytorch
  resnet_descriptors )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/resnet_augmentation.py
  sprokit/processes/pytorch
  resnet_augmentation )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/alexnet_descriptors.py
  sprokit/processes/pytorch
  alexnet_descriptors )

add_subdirectory( utils )
