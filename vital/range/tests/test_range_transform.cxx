/*ckwg +29
 * Copyright 2018-2019 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
