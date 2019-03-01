/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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
