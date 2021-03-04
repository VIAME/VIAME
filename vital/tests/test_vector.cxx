// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test core vector functionality
 */

#include <vital/types/vector.h>

#include <gtest/gtest.h>

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(vector, construct_2d)
{
  kwiver::vital::vector_2d v2d{ 10.0, 33.3 };
  EXPECT_EQ( 10.0, v2d.x() );
  EXPECT_EQ( 33.3, v2d.y() );

  kwiver::vital::vector_2f v2f{ 5.0f, 4.5f };
  EXPECT_EQ( 5.0f, v2f.x() );
  EXPECT_EQ( 4.5f, v2f.y() );
}

// ----------------------------------------------------------------------------
TEST(vector, construct_3d)
{
  kwiver::vital::vector_3d v3d{ 10.0, 33.3, 12.1 };
  EXPECT_EQ( 10.0, v3d.x() );
  EXPECT_EQ( 33.3, v3d.y() );
  EXPECT_EQ( 12.1, v3d.z() );

  kwiver::vital::vector_3f v3f{ 5.0f, 4.5f, -6.3f };
  EXPECT_EQ( 5.0f, v3f.x() );
  EXPECT_EQ( 4.5f, v3f.y() );
  EXPECT_EQ( -6.3f, v3f.z() );
}

// ----------------------------------------------------------------------------
TEST(vector, construct_4d)
{
  kwiver::vital::vector_4d v4d{ 10.0, 33.3, 12.1, 0.0 };
  EXPECT_EQ( 10.0, v4d.x() );
  EXPECT_EQ( 33.3, v4d.y() );
  EXPECT_EQ( 12.1, v4d.z() );
  EXPECT_EQ( 0.0, v4d.w() );

  kwiver::vital::vector_4f v4f{ 5.0f, 4.5f, -6.3f, 100.0f };
  EXPECT_EQ( 5.0f, v4f.x() );
  EXPECT_EQ( 4.5f, v4f.y() );
  EXPECT_EQ( -6.3f, v4f.z() );
  EXPECT_EQ( 100.0f, v4f.w() );
}
