// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test range filter
 */

#include "test_values.h"

#include <vital/range/filter.h>

#include <gtest/gtest.h>

#include <set>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(range_filter, empty)
{
  auto const empty_set = std::set< int >{};
  auto my_filter = []( int ){ return true; };

  int counter = 0;
  for ( auto x : empty_set | range::filter( my_filter ) )
  {
    static_cast< void >( x );
    ++counter;
  }

  EXPECT_EQ( 0, counter );
}

// ----------------------------------------------------------------------------
TEST(range_filter, always_true)
{
  auto my_filter = []( int ){ return true; };

  int counter = 0;
  for ( auto x : test_values | range::filter( my_filter ) )
  {
    static_cast< void >( x );
    ++counter;
  }

  EXPECT_EQ( 32, counter );
}

// ----------------------------------------------------------------------------
TEST(range_filter, always_false)
{
  auto my_filter = []( int ){ return false; };

  int counter = 0;
  for ( auto x : test_values | range::filter( my_filter ) )
  {
    static_cast< void >( x );
    ++counter;
  }

  EXPECT_EQ( 0, counter );
}

// ----------------------------------------------------------------------------
TEST(range_filter, specific_value)
{
  auto my_filter = []( int i ){ return i != 9; };

  int counter = 0;
  for ( auto x : test_values | range::filter( my_filter ) )
  {
    EXPECT_NE( 9, x ) << "At iteration " << counter;
    ++counter;
  }

  EXPECT_EQ( 29, counter );
}

// ----------------------------------------------------------------------------
TEST(range_filter, no_match)
{
  auto my_filter = []( int i ){ return i == 7; };

  int counter = 0;
  for ( auto x : test_values | range::filter( my_filter ) )
  {
    EXPECT_EQ( 7, x ) << "At iteration " << counter;
    ++counter;
  }

  EXPECT_EQ( 0, counter );
}

// ----------------------------------------------------------------------------
TEST(range_filter, evens)
{
  auto my_filter = []( int i ){ return ( i & 1 ) == 0; };

  int counter = 0;
  for ( auto x : test_values | range::filter( my_filter ) )
  {
    EXPECT_EQ( 0, ( x % 2 ) ) << "At iteration " << counter;
    ++counter;
  }

  EXPECT_EQ( 17, counter );
}

// ----------------------------------------------------------------------------
TEST(range_filter, odds)
{
  auto my_filter = []( int i ){ return ( i & 1 ) != 0; };

  int counter = 0;
  for ( auto x : test_values | range::filter( my_filter ) )
  {
    EXPECT_NE( 0, ( x % 2 ) ) << "At iteration " << counter;
    ++counter;
  }

  EXPECT_EQ( 15, counter );
}
