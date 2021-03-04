// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test range transform
 */

#include "test_values.h"

#include <vital/range/transform.h>

#include <gtest/gtest.h>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(range_transform, basic)
{
  auto my_transform = []( int x ){
    return static_cast< double >( x ) / 16.0;
  };

  int counter = 0;
  auto sum = 0.0;
  for ( auto x : test_values | range::transform( my_transform ) )
  {
    sum += x;

    auto const t = static_cast< double >( test_values[ counter ] );
    EXPECT_EQ( t / 16.0, x ) << "At iteration " << counter;

    ++counter;
  }

  EXPECT_EQ( 32, counter );
  EXPECT_EQ( 13.5625, sum );
}

// ----------------------------------------------------------------------------
TEST(range_transform, transitive)
{
  auto my_transform_1 = []( int x ){
    return x + 1;
  };
  auto my_transform_2 = []( int x ){
    return static_cast< double >( x ) / 4.0;
  };

  int counter = 0;
  auto sum = 0.0;
  for ( auto x : test_values | range::transform( my_transform_1 )
                             | range::transform( my_transform_2 ) )
  {
    sum += x;

    auto const t = static_cast< double >( test_values[ counter ] );
    EXPECT_EQ( ( t + 1.0 ) / 4.0, x ) << "At iteration " << counter;

    ++counter;
  }

  EXPECT_EQ( 32, counter );
  EXPECT_EQ( 62.25, sum );
}

// ----------------------------------------------------------------------------
TEST(range_transform, temporary)
{
  class temporary_vector : public std::vector< int >
  {
  public:
    using std::vector< int >::vector;
    ~temporary_vector()
    {
      for ( auto& x : *this )
      {
        x = 0;
      }
    }
  };

  auto make_temporary = []{
    auto out = temporary_vector{};
    std::copy( std::begin( test_values ), std::end( test_values ),
               std::back_inserter( out ) );
    return out;
  };

  auto my_transform = []( int x ){
    return static_cast< double >( x ) / 16.0;
  };

  int counter = 0;
  auto sum = 0.0;
  for ( auto x : make_temporary() | range::transform( my_transform ) )
  {
    sum += x;

    auto const t = static_cast< double >( test_values[ counter ] );
    EXPECT_EQ( t / 16.0, x ) << "At iteration " << counter;

    ++counter;
  }

  EXPECT_EQ( 32, counter );
  EXPECT_EQ( 13.5625, sum );
}
