// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test VXL bundle adjustment functionality
 */

#include <arrows/ocv/bounding_box.h>

#include <gtest/gtest.h>

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(bounding_box, convert_bb2ocv)
{
  kwiver::vital::bounding_box<double> bbox( 1, 3, 10, 34 );
  auto const& vbox = kwiver::arrows::ocv::convert( bbox );

  EXPECT_EQ( bbox.min_x(), vbox.x );
  EXPECT_EQ( bbox.min_y(), vbox.y );
  EXPECT_EQ( bbox.width(), vbox.width );
  EXPECT_EQ( bbox.height(), vbox.height );
}

// ----------------------------------------------------------------------------
TEST(bounding_box, convert_ocv2bb)
{
  auto vbox= cv::Rect( 1, 3, 10, 34 );
  auto const& bbox = kwiver::arrows::ocv::convert<double>( vbox );

  EXPECT_EQ( vbox.x, bbox.min_x() );
  EXPECT_EQ( vbox.y, bbox.min_y() );
  EXPECT_EQ( vbox.width, bbox.width() );
  EXPECT_EQ( vbox.height, bbox.height() );
}
