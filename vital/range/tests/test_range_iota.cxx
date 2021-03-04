// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test range iota
 */

#include <vital/range/iota.h>

#include <gtest/gtest.h>

#include <memory>
#include <vector>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(range_iota, empty)
{
  auto test_values = std::vector< int >{};

  auto counter = int{ 0 };
  for ( auto const x : range::iota( int{ 0 } ) )
  {
    test_values.push_back( x );
    ++counter;
  }

  EXPECT_EQ( 0, counter );
  EXPECT_TRUE( test_values.empty() );
}

// ----------------------------------------------------------------------------
TEST(range_iota, basic)
{
  auto test_values = std::vector< int >{};

  auto counter = int{ 0 };
  auto accumulator = int{ 0 };
  for ( auto const x : range::iota( int{ 5 } ) )
  {
    test_values.push_back( x );
    ++counter;
    accumulator += x;
  }

  EXPECT_EQ( 5, counter );
  EXPECT_EQ( 10, accumulator );

  ASSERT_EQ( 5, test_values.size() );
  EXPECT_EQ( 0, test_values[ 0 ] );
  EXPECT_EQ( 1, test_values[ 1 ] );
  EXPECT_EQ( 2, test_values[ 2 ] );
  EXPECT_EQ( 3, test_values[ 3 ] );
  EXPECT_EQ( 4, test_values[ 4 ] );
}

// ----------------------------------------------------------------------------
TEST(range_iota, limit)
{
  constexpr auto limit = int{ 1 } << 20;

  auto counter = int{ 0 };
  for ( auto const x : range::iota( limit ) )
  {
    static_cast< void >( x );
    ++counter;
  }

  EXPECT_EQ( limit, counter );
}
