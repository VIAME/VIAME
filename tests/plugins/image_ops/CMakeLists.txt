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
