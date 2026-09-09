# VIAME image_ops Tests

project( test_viame_image_ops )

include( kwiver-test-setup )

kwiver_discover_gtests( viame image_ops
  LIBRARIES viame_image_ops
            kwiver::vital
            GTest::gtest )
