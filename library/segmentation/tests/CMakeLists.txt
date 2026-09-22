# VIAME segmentation tests

project( test_viame_segmentation )

include( viame-test-setup )

if( VIAME_ENABLE_PYTHON )
  # `python -m viame.core.interactive_service`, the command DIVE's desktop
  # client starts, forwarded to `viame.segmentation.interactive_service`
  # since P2-T07 emptied `viame.core`.
  viame_add_test(
    NAME    "unit:segmentation:dive_service_path"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_dive_service_path.py"
    LABELS  UNIT
    TIMEOUT 300
    SOURCE_SETUP
  )
endif()

# SAM3, from `tests/plugins/pytorch` in P2-T09. `test_sam3_interactive.py`
# came too, unregistered as it was.
if( VIAME_ENABLE_PYTHON AND VIAME_ENABLE_PYTORCH )
  viame_add_test(
    NAME    "unit:segmentation:sam3_utilities"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py" -k TestSAM3Utilities
    LABELS  UNIT PYTORCH SAM3
    TIMEOUT 300
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
  )
  viame_add_test(
    NAME    "unit:segmentation:sam3_model_manager"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py" -k TestSAM3ModelManager
    LABELS  UNIT PYTORCH SAM3
    TIMEOUT 300
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
  )
  viame_add_test(
    NAME    "unit:segmentation:sam3_refiner"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py" -k TestSAM3Refiner
    LABELS  UNIT PYTORCH SAM3
    TIMEOUT 300
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
  )
  viame_add_test(
    NAME    "integration:segmentation:sam3_pipelines"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py" -k TestSAM3Pipelines
    LABELS  INTEGRATION PYTORCH SAM3
    TIMEOUT 900
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
  )
  viame_add_test(
    NAME    "integration:segmentation:sam3_text_queries"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py" -k TestSAM3TextQueries
    LABELS  INTEGRATION PYTORCH SAM3
    TIMEOUT 600
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
  )
  viame_add_test(
    NAME    "integration:segmentation:sam3_point_click"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py" -k TestSAM3PointClickSegmentation
    LABELS  INTEGRATION PYTORCH SAM3
    TIMEOUT 600
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
  )
  viame_add_test(
    NAME    "unit:segmentation:sam3_registration"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py" -k TestSAM3AlgorithmRegistration
    LABELS  UNIT PYTORCH SAM3
    TIMEOUT 300
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
  )
  viame_add_test(
    NAME    "integration:segmentation:sam3_integration"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py" -k TestSAM3Integration
    LABELS  INTEGRATION PYTORCH SAM3
    TIMEOUT 900
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
  )
  viame_add_test(
    NAME    "integration:segmentation:sam3_all"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_sam3.py"
    LABELS  INTEGRATION PYTORCH SAM3
    TIMEOUT 1800
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
  )
endif()

# From `tests/plugins/core` upstream, which lite dissolved. They load
# `segmentation_utils` by path, repointed to `library/segmentation`.
if( VIAME_ENABLE_PYTHON )
  foreach( case IN ITEMS multipolygon_keypoints segmentation_line_scale
                         segmentation_prompt_instances )
    viame_add_test(
      NAME    "unit:segmentation:${case}"
      TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_${case}.py"
      LABELS  UNIT
      TIMEOUT 300
      SOURCE_SETUP
    )
  endforeach()
endif()
