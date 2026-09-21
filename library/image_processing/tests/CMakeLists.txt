# VIAME image_processing tests

project( test_viame_image_processing )

include( viame-test-setup )

if( VIAME_ENABLE_PYTHON )
  # SIFT, SURF, the FLANN matcher and the two estimators are python since
  # P7-T04. The registry dump cannot read a python implementation's config,
  # so their config contract is held here rather than by compare_registry.
  viame_add_test(
    NAME    "unit:image_processing:ocv_features"
    TARGET  "${CMAKE_CURRENT_SOURCE_DIR}/test_ocv_features.py"
    LABELS  UNIT
    TIMEOUT 600
    SOURCE_SETUP
  )
endif()

# The SURF port against what OpenCV computes. The reference is recorded from a
# cv2 built with the non-free modules, which this branch's wheel is not -- see
# tests/golden/surf/record_from_opencv.py.
viame_discover_gtests( viame surf
  LIBRARIES viame_image_processing
            viame_image_io
            kwiver::vital
            GTest::gtest )

target_compile_definitions( test-viame-surf PRIVATE
  "VIAME_GOLDEN_SURF_DIR=\"${VIAME_SOURCE_DIR}/tests/golden/surf\"" )
