###
# Loadable algorithms and processes

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

add_subdirectory( utils )
