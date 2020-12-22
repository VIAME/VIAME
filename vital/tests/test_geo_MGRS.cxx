// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test geo_MGRS functionality
 */

#include <gtest/gtest.h>

#include <vital/types/geo_MGRS.h>

#include <sstream>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(geo_MGRS, default_constructor)
{
  geo_MGRS gm;
  EXPECT_TRUE(  gm.is_empty() );
  EXPECT_FALSE( gm.is_valid() );
}

// ----------------------------------------------------------------------------
TEST(geo_MGRS, constructor_str)
{
  geo_MGRS gm( "test_str12345" );
  EXPECT_FALSE( gm.is_empty() );
  EXPECT_TRUE(  gm.is_valid() );
}

// ----------------------------------------------------------------------------
TEST(geo_MGRS, coord)
{
  geo_MGRS gm;
  geo_MGRS gm_cpy = gm.set_coord( "test_str12345" );

  EXPECT_EQ( gm.coord(), "test_str12345" );
  EXPECT_EQ( gm_cpy.coord(), "test_str12345" );
  EXPECT_FALSE( gm.is_empty() );
  EXPECT_TRUE(  gm.is_valid() );
  EXPECT_FALSE( gm_cpy.is_empty() );
  EXPECT_TRUE(  gm_cpy.is_valid() );

  gm_cpy = gm.set_coord( "another_test_str" );
  EXPECT_EQ( gm.coord(), "another_test_str" );
  EXPECT_EQ( gm_cpy.coord(), "another_test_str" );
  EXPECT_FALSE( gm.is_empty() );
  EXPECT_TRUE(  gm.is_valid() );
  EXPECT_FALSE( gm_cpy.is_empty() );
  EXPECT_TRUE(  gm_cpy.is_valid() );

  // Test return to empty
  gm_cpy = gm.set_coord( "" );
  EXPECT_EQ( gm.coord(), "" );
  EXPECT_EQ( gm_cpy.coord(), "" );
  EXPECT_TRUE(  gm.is_empty() );
  EXPECT_FALSE( gm.is_valid() );
  EXPECT_TRUE(  gm_cpy.is_empty() );
  EXPECT_FALSE( gm_cpy.is_valid() );
}

// ----------------------------------------------------------------------------
TEST(geo_MGRS, comparisons)
{
  geo_MGRS gm1, gm2;
  EXPECT_TRUE(  gm1 == gm2 );
  EXPECT_FALSE( gm1 != gm2 );

  // Check copies are equal
  geo_MGRS gm1_cpy = gm1.set_coord( "test_str12345" );
  EXPECT_FALSE( gm1 == gm2 );
  EXPECT_TRUE(  gm1 != gm2 );
  EXPECT_TRUE(  gm1_cpy == gm1 );
  EXPECT_FALSE( gm1_cpy != gm1 );
  EXPECT_FALSE( gm1_cpy == gm2 );
  EXPECT_TRUE(  gm1_cpy != gm2 );

  gm2.set_coord( "test_str12345" );
  EXPECT_TRUE(  gm1 == gm2 );
  EXPECT_FALSE( gm1 != gm2 );
  EXPECT_TRUE(  gm1_cpy == gm1 );
  EXPECT_FALSE( gm1_cpy != gm1 );
  EXPECT_TRUE(  gm1_cpy == gm2 );
  EXPECT_FALSE( gm1_cpy != gm2 );
}

// ----------------------------------------------------------------------------
TEST(geo_MGRS, insert_operator_empty)
{
  geo_MGRS gm;
  std::stringstream s;
  s << gm;
  EXPECT_EQ( s.str(), "[MGRS: ]" );
}

// ----------------------------------------------------------------------------
TEST(geo_MGRS, insert_operator)
{
  geo_MGRS gm( "test_str12345" );
  std::stringstream s;
  s << gm;
  EXPECT_EQ( s.str(), "[MGRS: test_str12345]" );
}
