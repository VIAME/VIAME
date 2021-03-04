// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test point_2 functionality
 */

#include <vital/types/bounding_box.h>

#include <gtest/gtest.h>

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(bounding_box, construct_bbox_i)
{
  kwiver::vital::bounding_box_i::vector_type tl{ 12, 23 };
  kwiver::vital::bounding_box_i::vector_type  br{ 200, 223 };
  kwiver::vital::bounding_box_i bb{ tl, br };

  EXPECT_EQ( tl, bb.upper_left() );
  EXPECT_EQ( br, bb.lower_right() );
}

// ----------------------------------------------------------------------------
TEST(bounding_box, construct_bbox_d)
{
  kwiver::vital::bounding_box_d::vector_type tl{ 12, 23 };
  kwiver::vital::bounding_box_d::vector_type br{ 12 + 111, 23 + 222 };
  kwiver::vital::bounding_box_d bb1{ tl, 111, 222 };

  EXPECT_EQ( tl, bb1.upper_left() );
  EXPECT_EQ( br, bb1.lower_right() );
}

// ----------------------------------------------------------------------------
TEST(bounding_box, translate_bbox_d)
{
  kwiver::vital::bounding_box_d::vector_type tl{ 12, 23 };
  kwiver::vital::bounding_box_d::vector_type br{ 200, 223 };
  kwiver::vital::bounding_box_d::vector_type t{ 20, 10 };
  kwiver::vital::bounding_box_d bb{ tl, br };

  kwiver::vital::translate( bb, t );

  EXPECT_EQ( 32, bb.upper_left().x() );
  EXPECT_EQ( 33, bb.upper_left().y() );
  EXPECT_EQ( 220, bb.lower_right().x() );
  EXPECT_EQ( 233, bb.lower_right().y() );
}

// ----------------------------------------------------------------------------
TEST(bounding_box, intersection_bbox_d)
{
  kwiver::vital::bounding_box_d::vector_type tl{ 12, 23 };
  kwiver::vital::bounding_box_d::vector_type br{ 200, 223 };
  kwiver::vital::bounding_box_d::vector_type t{ 120, 110 };
  kwiver::vital::bounding_box_d bb1{ tl, br };

  kwiver::vital::bounding_box_d bb2 = bb1;
  kwiver::vital::translate( bb2, t );
  kwiver::vital::bounding_box_d bbi = kwiver::vital::intersection( bb1, bb2 );

  EXPECT_EQ( 132, bbi.upper_left().x() );
  EXPECT_EQ( 133, bbi.upper_left().y() );
  EXPECT_EQ( 200, bbi.lower_right().x() );
  EXPECT_EQ( 223, bbi.lower_right().y() );
}

// ----------------------------------------------------------------------------
TEST(bounding_box, comparisons)
{
  kwiver::vital::bounding_box_d::vector_type tl1{ 12, 23 };
  kwiver::vital::bounding_box_d::vector_type br1{ 200, 223 };
  kwiver::vital::bounding_box_d::vector_type tl2{ 10, 15 };
  kwiver::vital::bounding_box_d::vector_type br2{ 120, 110 };

  kwiver::vital::bounding_box_d bb1{ tl1, br1 };
  kwiver::vital::bounding_box_d bb1_clone = bb1;
  kwiver::vital::bounding_box_d bb2{ tl2, br2 };

  // Check ==
  EXPECT_TRUE(  bb1 == bb1_clone );
  EXPECT_FALSE( bb1 == bb2 );
  EXPECT_FALSE( bb1_clone == bb2 );

  // Check !=
  EXPECT_FALSE(  bb1 != bb1_clone );
  EXPECT_TRUE( bb1 != bb2 );
  EXPECT_TRUE( bb1_clone != bb2 );
}
