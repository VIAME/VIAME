# VIAME image_ops Tests

project( test_viame_image_ops )

include( kwiver-test-setup )

kwiver_discover_gtests( viame image_ops
  LIBRARIES viame_image_ops
            kwiver::vital
            GTest::gtest )

# The A/B against VXL was run while VXL was still built and is recorded in
# design/STATUS.md; what remains are the values that are known analytically
kwiver_discover_gtests( viame homography_overlap
  LIBRARIES viame_image_ops
            kwiver::vital
            GTest::gtest )

# The colour conversions, against a recording of what OpenCV computed for the
# same pixels. Reading a file rather than calling OpenCV is what lets the
# test outlive it.
kwiver_discover_gtests( viame color
  LIBRARIES viame_image_ops
            kwiver::vital
            GTest::gtest )

# The convolution kernels, against the same recording.
kwiver_discover_gtests( viame filter
  LIBRARIES viame_image_ops
            kwiver::vital
            GTest::gtest )

# The geometric transforms, against the same recording.
kwiver_discover_gtests( viame warp
  LIBRARIES viame_image_ops
            kwiver::vital
            GTest::gtest )

# Histograms and the equalisations, against the same recording.
kwiver_discover_gtests( viame histogram
  LIBRARIES viame_image_ops
            kwiver::vital
            GTest::gtest )

# Connected components and contours, against the same recording.
kwiver_discover_gtests( viame contours
  LIBRARIES viame_image_ops
            kwiver::vital
            GTest::gtest )

# Drawing, template matching and concatenation, against the same recording.
kwiver_discover_gtests( viame draw
  LIBRARIES viame_image_ops
            kwiver::vital
            GTest::gtest )

foreach( test IN ITEMS color filter warp histogram contours draw )
  target_compile_definitions( test-viame-${test} PRIVATE
    "VIAME_GOLDEN_IMAGE_OPS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/../../golden/image_ops\"" )
endforeach()
