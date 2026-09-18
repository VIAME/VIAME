# VIAME image_kernels Tests

project( test_viame_image_kernels )

include( viame-test-setup )

viame_discover_gtests( viame image_kernels
  LIBRARIES viame_image_kernels
            kwiver::vital
            GTest::gtest )

# The A/B against VXL was run while VXL was still built and is recorded in
# design/STATUS.md; what remains are the values that are known analytically
viame_discover_gtests( viame homography_overlap
  LIBRARIES viame_image_kernels
            kwiver::vital
            GTest::gtest )

# The colour conversions, against a recording of what OpenCV computed for the
# same pixels. Reading a file rather than calling OpenCV is what lets the
# test outlive it.
viame_discover_gtests( viame color
  LIBRARIES viame_image_kernels
            kwiver::vital
            GTest::gtest )

# The convolution kernels, against the same recording.
viame_discover_gtests( viame filter
  LIBRARIES viame_image_kernels
            kwiver::vital
            GTest::gtest )

# The geometric transforms, against the same recording.
viame_discover_gtests( viame warp
  LIBRARIES viame_image_kernels
            kwiver::vital
            GTest::gtest )

# Histograms and the equalisations, against the same recording.
viame_discover_gtests( viame histogram
  LIBRARIES viame_image_kernels
            kwiver::vital
            GTest::gtest )

# Connected components and contours, against the same recording.
viame_discover_gtests( viame contours
  LIBRARIES viame_image_kernels
            kwiver::vital
            GTest::gtest )

# Drawing, template matching and concatenation, against the same recording.
viame_discover_gtests( viame draw
  LIBRARIES viame_image_kernels
            kwiver::vital
            GTest::gtest )

foreach( test IN ITEMS color filter warp histogram contours draw )
  target_compile_definitions( test-viame-${test} PRIVATE
    "VIAME_GOLDEN_IMAGE_KERNELS_DIR=\"${VIAME_SOURCE_DIR}/tests/golden/image_kernels\"" )
endforeach()

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

viame_discover_gtests( viame convert_polygons_to_mask
  LIBRARIES ${moved_core_test_libraries} )

viame_discover_gtests( viame windowed_refiner_utils
  LIBRARIES ${moved_core_test_libraries} )
