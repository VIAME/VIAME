// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test VXL bundle adjustment functionality
 */

#include <arrows/vxl/bounding_box.h>

#include <gtest/gtest.h>

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(bounding_box, convert_bb2vgl)
{
  kwiver::vital::bounding_box<double> bbox( 1.1, 3.4, 10.12, 34.45 );
  vgl_box_2d<double> vbox = kwiver::arrows::vxl::convert( bbox );

  EXPECT_EQ( bbox.min_x(), vbox.min_x() );
  EXPECT_EQ( bbox.min_y(), vbox.min_y() );
  EXPECT_EQ( bbox.max_x(), vbox.max_x() );
  EXPECT_EQ( bbox.max_y(), vbox.max_y() );
}

// ----------------------------------------------------------------------------
TEST(bounding_box, convert_vgl2bb)
{
  vgl_box_2d<double> vbox( 1.1, 3.4, 10.12, 34.45 );
  kwiver::vital::bounding_box<double> bbox = kwiver::arrows::vxl::convert( vbox );

  EXPECT_EQ( vbox.min_x(), bbox.min_x() );
  EXPECT_EQ( vbox.min_y(), bbox.min_y() );
  EXPECT_EQ( vbox.max_x(), bbox.max_x() );
  EXPECT_EQ( vbox.max_y(), bbox.max_y() );
}
