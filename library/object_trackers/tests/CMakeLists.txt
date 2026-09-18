# VIAME object tracker tests

project( test_viame_object_trackers )

include( viame-test-setup )

if( VIAME_ENABLE_PYTHON )
  # The Kalman filter the ByteTrack-family trackers share since P2-T06, held
  # to a recording of the four copies it replaced.
  viame_add_test(
    NAME    "unit:object_trackers:kalman"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_kalman.py"
    LABELS  UNIT
    TIMEOUT 300
    SOURCE_SETUP
  )
endif()

# From `tests/plugins/core` in P2-T09. Linked as they were there, against the
# libraries the core plugin's tests shared, so moving them changed nothing
# but where they are.
set( moved_core_test_libraries
  viame_utilities
  viame_image_kernels
  viame_file_io
  viame_training
  viame_evaluation
  kwiver::vital
  kwiver::vital_algo
  GTest::gtest_main )

viame_discover_gtests( viame resample_object_tracks
  LIBRARIES ${moved_core_test_libraries} )

if( VIAME_ENABLE_PYTHON )
  # Every tracker trainer loads and survives a configuration round trip. Cheap
  # enough to be worth running whatever else is enabled; trainers whose
  # subsystem is not built skip themselves.
  viame_add_test(
    NAME    "unit:object_trackers:tracker_trainers_load"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_tracker_trainers_load.py"
    LABELS  UNIT CORE PYTORCH
    TIMEOUT 300
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
  )
endif()

# SRNN and the torchvision descriptor processes, from `tests/plugins/pytorch`:
# one module, since it imports the SRNN models at module level.
if( VIAME_ENABLE_PYTHON AND VIAME_ENABLE_PYTORCH )
  viame_add_test(
    NAME    "unit:object_trackers:srnn_tracker"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_processes.py" -k test_srnn_tracker
    LABELS  UNIT PYTORCH
    TIMEOUT 600
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
  )

  if( VIAME_ENABLE_PYTORCH-VISION )
    viame_add_test(
      NAME    "unit:object_trackers:alexnet_descriptors"
      TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_processes.py" -k test_alexnet_descriptors
      LABELS  UNIT PYTORCH
      TIMEOUT 600
      SOURCE_SETUP
      WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
    )
    viame_add_test(
      NAME    "unit:object_trackers:resnet_descriptors"
      TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_processes.py" -k test_resnet_descriptors
      LABELS  UNIT PYTORCH
      TIMEOUT 600
      SOURCE_SETUP
      WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
    )
    viame_add_test(
      NAME    "unit:object_trackers:desc_augmentation"
      TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_processes.py" -k test_desc_augmentation
      LABELS  UNIT PYTORCH
      TIMEOUT 600
      SOURCE_SETUP
      WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
    )
  endif()
endif()
