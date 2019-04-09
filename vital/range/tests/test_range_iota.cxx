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
