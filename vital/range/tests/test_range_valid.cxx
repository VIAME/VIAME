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
 * \brief test range validity-filter
 */

#include <vital/range/valid.h>

#include <gtest/gtest.h>

#include <memory>
#include <vector>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(range_valid, empty)
{
  auto test_values = std::vector< int >{};

  auto counter = int{ 0 };
  for ( auto x : test_values | range::valid )
  {
    static_cast< void >( x );
    ++counter;
  }

  EXPECT_EQ( 0, counter );
}

// ----------------------------------------------------------------------------
TEST(range_valid, none)
{
  auto test_values = std::vector< bool >{ false, false };

  auto counter = int{ 0 };
  for ( auto x : test_values | range::valid )
  {
    static_cast< void >( x );
    ++counter;
  }

  EXPECT_EQ( 0, counter );
}

// ----------------------------------------------------------------------------
TEST(range_valid, basic)
{
  auto test_values = std::vector< std::shared_ptr< int > >{
    std::shared_ptr< int >{ nullptr },
    std::shared_ptr< int >{ new int{ 1 } },
    std::shared_ptr< int >{ nullptr },
    std::shared_ptr< int >{ new int{ 2 } },
    std::shared_ptr< int >{ new int{ 3 } },
    std::shared_ptr< int >{ nullptr },
    std::shared_ptr< int >{ nullptr },
    std::shared_ptr< int >{ new int{ 4 } },
    std::shared_ptr< int >{ new int{ 5 } },
    std::shared_ptr< int >{ nullptr }
  };

  auto accumulator = int{ 0 };
  for ( auto p : test_values | range::valid )
  {
    accumulator += *p;
  }

  EXPECT_EQ( 15, accumulator );
}

// ----------------------------------------------------------------------------
TEST(range_valid, mutating)
{
  auto test_values = std::vector< int >{ 1, 2, 3, 4, 5 };

  for ( auto& x : test_values | range::valid )
  {
    if ( x == 3 )
    {
      x = 42;
    }
  }

  EXPECT_EQ( 42, test_values[2] );
}
