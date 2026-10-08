# VIAME object_detectors tests

project( test_viame_object_detectors )

include( viame-test-setup )

if( VIAME_ENABLE_PYTHON )
  # `hough_circle` is python since P7-T04, and the registry dump cannot read
  # a python implementation's config, so its config contract is held here.
  # `stereo_algos`, which three shipped pipelines reach through
  # `gmm_motion_detector` and which had no tests of any kind until it came off
  # cv2 in P7.
  # CRITICAL rather than UNIT. This branch takes torch from the locks, so it
  # should always pass -- the point is that it says so. A Windows source
  # build linking two OpenMP runtimes returns garbage for every threaded CPU
  # result above ATen's GRAIN_SIZE, and the v0.23.5 desktop binaries shipped
  # that way: RF-DETR trained to completion on blank images while a training
  # test watched and passed. The sizes bracket the boundary at 104 and 105.
  viame_add_test(
    NAME    "critical:object_detectors:threaded_arithmetic"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_torch_parallel_arithmetic.py"
    LABELS  CRITICAL UNIT
    TIMEOUT 120
    SOURCE_SETUP
  )

  viame_add_test(
    NAME    "unit:object_detectors:stereo_algos"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_stereo_algos.py"
    LABELS  UNIT
    TIMEOUT 300
    SOURCE_SETUP
  )

  viame_add_test(
    NAME    "unit:object_detectors:hough_circle"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_hough_circle_detector.py"
    LABELS  UNIT
    TIMEOUT 600
    SOURCE_SETUP
  )

  viame_add_test(
    NAME    "unit:object_detectors:heat_map_closing"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_detect_heat_map_closing.py"
    LABELS  UNIT
    TIMEOUT 600
    SOURCE_SETUP
  )
endif()

# From `tests/plugins/pytorch` in P2-T09: the netharn and RF-DETR trainer
# sources, read by path. `test_feature_device.py` came too, unregistered.
if( VIAME_ENABLE_PYTHON AND VIAME_ENABLE_PYTORCH )
  # netharn's stereo rectification, which had no test at all before it was
  # ported off cv2 -- so this is the check that did not exist rather than one
  # of something already covered.
  viame_add_test(
    NAME    "unit:object_detectors:netharn_stereo"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_netharn_stereo.py"
    LABELS  UNIT
    TIMEOUT 600
    SOURCE_SETUP
  )

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

if( VIAME_ENABLE_PYTHON )
  viame_add_test(
    NAME "unit:object_detectors:rfdetr_seed_weights"
    TARGET "${CMAKE_CURRENT_SOURCE_DIR}/test_rfdetr_seed_weights.py"
    LABELS UNIT TIMEOUT 300 SOURCE_SETUP )
endif()

if( VIAME_ENABLE_PYTHON )
  viame_add_test( NAME "unit:object_detectors:ollama_vlm"
    TARGET "${CMAKE_CURRENT_SOURCE_DIR}/test_ollama_vlm.py"
    LABELS UNIT TIMEOUT 120 SOURCE_SETUP )
endif()
