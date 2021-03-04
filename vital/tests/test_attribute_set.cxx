// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test attribute_set functionality
 */

#include <vital/attribute_set.h>

#include <gtest/gtest.h>

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(attribute_set, api)
{
  kwiver::vital::attribute_set set;

  EXPECT_EQ( 0, set.size() );
  EXPECT_TRUE( set.empty() );

  set.add( "first", 1 );
  EXPECT_EQ( 1, set.size() );
  EXPECT_FALSE( set.empty() );

  // Replace entry
  set.add( "first", (double) 3.14159 );
  EXPECT_EQ( 1, set.size() );
  EXPECT_FALSE( set.empty() );
  EXPECT_TRUE( set.has( "first" ) );
  EXPECT_FALSE( set.has( "the-first" ) );

  set.add( "second", 42 );
  EXPECT_EQ( 2, set.size() );
  EXPECT_FALSE( set.empty() );
  EXPECT_TRUE( set.has( "first" ) );
  EXPECT_TRUE( set.has( "second" ) );

  EXPECT_EQ( 42, set.get<int>( "second" ) );
  EXPECT_TRUE( set.is_type<int>( "second") );

  // Test returning wrong type, exception
  EXPECT_THROW(
    set.get<double>( "second" ),
    kwiver::vital::bad_any_cast );

  // Test data() accessor
  EXPECT_THROW(
    set.data( "does-not-exist" ),
    kwiver::vital::attribute_set_exception );

  // Test iterators
  for ( auto it = set.begin(); it != set.end(); ++it )
  {
    std::cout << it->first << std::endl;
  }
}
