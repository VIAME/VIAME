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
 * \brief test range sliding view
 */

#include "test_values.h"

#include <vital/range/sliding.h>

#include <gtest/gtest.h>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(range_sliding, basic)
{
  int const expected_values[] = {
    0, 39, 2, 21, 44, 48, 9, -2, 53, 113,
    89, 159, 52, 17, 3, -9, -12, 106, 169, 129,
    85, 98, 16, 9, 8, 9, 70, 48, 60,
  };

  int counter = 0;
  for ( auto x : test_values | range::sliding< 4 >() )
  {
    static_assert( x.size() == 4, "unexpected window size" );

    ASSERT_LT( counter, 29 );

    auto const y = x[0] + ( x[1] * x[2] ) - x[3];
    EXPECT_EQ( expected_values[ counter ], y ) << "At iteration " << counter;

    ++counter;
  }

  EXPECT_EQ( 29, counter );
}

// ----------------------------------------------------------------------------
TEST(range_sliding, same_size_as_input)
{
  int counter = 0;
  for ( auto x : test_values | range::sliding< 32 >() )
  {
    static_assert( x.size() == 32, "unexpected window size" );

    int i = 0;
    for ( auto y : x )
    {
      EXPECT_EQ( test_values[ i ], y );
      ++i;
    }

    ++counter;
  }

  EXPECT_EQ( 1, counter );
}

// ----------------------------------------------------------------------------
TEST(range_sliding, input_too_small)
{
  int counter = 0;
  for ( auto x : test_values | range::sliding< 33 >() )
  {
    static_assert( x.size() == 33, "unexpected window size" );
    ++counter;
  }

  EXPECT_EQ( 0, counter );
}
