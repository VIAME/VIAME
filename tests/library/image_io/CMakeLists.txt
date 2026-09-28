# VIAME image_io tests

project( test_viame_image_io )

include( viame-test-setup )

# The in-house codecs: what the formats say, on images small enough to reason
# about by hand. Agreement with OpenCV on real files is tests/reference/codecs.
viame_discover_gtests( viame codecs
  LIBRARIES viame_image_io
            kwiver::vital
            GTest::gtest )

if( VIAME_ENABLE_PYTHON )
  viame_add_test( NAME "unit:image_io:display"
    TARGET "${CMAKE_CURRENT_SOURCE_DIR}/test_display.py"
    LABELS UNIT TIMEOUT 60 SOURCE_SETUP )
endif()
