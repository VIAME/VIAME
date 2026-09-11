# VIAME image_processing tests

project( test_viame_image_processing )

include( kwiver-test-setup )

if( VIAME_ENABLE_PYTHON )
  # SIFT, SURF, the FLANN matcher and the two estimators are python since
  # P7-T04. The registry dump cannot read a python implementation's config,
  # so their config contract is held here rather than by compare_registry.
  viame_add_pytest_test(
    NAME    "unit:image_processing:ocv_features"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_ocv_features.py"
    LABELS  UNIT
    TIMEOUT 600
    SOURCE_SETUP
  )
endif()
