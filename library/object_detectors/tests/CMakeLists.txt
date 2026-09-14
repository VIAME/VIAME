# VIAME object_detectors tests

project( test_viame_object_detectors )

include( viame-test-setup )

if( VIAME_ENABLE_PYTHON )
  # `hough_circle` is python since P7-T04, and the registry dump cannot read
  # a python implementation's config, so its config contract is held here.
  viame_add_test(
    NAME    "unit:object_detectors:hough_circle"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_hough_circle_detector.py"
    LABELS  UNIT
    TIMEOUT 600
    SOURCE_SETUP
  )
endif()

# From `tests/plugins/pytorch` in P2-T09: the netharn and RF-DETR trainer
# sources, read by path. `test_feature_device.py` came too, unregistered.
if( VIAME_ENABLE_PYTHON AND VIAME_ENABLE_PYTORCH )
  viame_add_test(
    NAME    "unit:object_detectors:netharn_peer_probe"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_netharn_peer_probe.py"
    LABELS  UNIT PYTORCH
    TIMEOUT 120
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
  )
  viame_add_test(
    NAME    "unit:object_detectors:netharn_rfdetr_training"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_netharn_rfdetr_training.py"
    LABELS  UNIT PYTORCH
    TIMEOUT 300
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
  )
  viame_add_test(
    NAME    "unit:object_detectors:rfdetr_multipolygon"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_rfdetr_multipolygon.py"
    LABELS  UNIT PYTORCH
    TIMEOUT 300
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
  )
endif()
