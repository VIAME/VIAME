# VIAME classifier tests

project( test_viame_classifiers )

include( viame-test-setup )

# The SLEAP keypoint trainer, launcher and refiner, from `tests/plugins/pytorch`
# in P2-T09. They import without kwiver, over stand-in packages.
if( VIAME_ENABLE_PYTHON AND VIAME_ENABLE_PYTORCH AND VIAME_ENABLE_PYTORCH-SLEAP )
  viame_add_test(
    NAME    "unit:classifiers:sleap_keypoints"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_sleap_keypoints.py"
    LABELS  UNIT PYTORCH SLEAP
    TIMEOUT 300
    SOURCE_SETUP
    WORKING_DIRECTORY "${VIAME_BUILD_INSTALL_PREFIX}"
  )
endif()
