/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
 * \brief core polygon class tests
 */

#include <vital/types/polygon.h>

#include <gtest/gtest.h>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

namespace {
static const polygon::point_t p1{ 10, 10 };
static const polygon::point_t p2{ 10, 50 };
static const polygon::point_t p3{ 50, 50 };
static const polygon::point_t p4{ 30, 30 };
}

// ----------------------------------------------------------------------------
TEST(polygon, default_constructor)
{
  polygon p;
  EXPECT_EQ( 0, p.num_vertices() );
}

// ----------------------------------------------------------------------------
TEST(polygon, construct_from_vector)
{
  std::vector<polygon::point_t> vec;

  vec.push_back( p1 );
  vec.push_back( p2 );
  vec.push_back( p3 );
  vec.push_back( p4 );

  polygon p( vec );
  EXPECT_EQ( 4, p.num_vertices() );
}


// ----------------------------------------------------------------------------
TEST(polygon, add_points)
{
  polygon p;

  p.push_back( p1 );
  ASSERT_EQ( 1, p.num_vertices() );

  p.push_back( p2 );
  ASSERT_EQ( 2, p.num_vertices() );

  p.push_back( p3 );
  ASSERT_EQ( 3, p.num_vertices() );

  p.push_back( p4 );
  ASSERT_EQ( 4, p.num_vertices() );

  EXPECT_EQ( p1, p.at( 0 ) );
  EXPECT_EQ( p2, p.at( 1 ) );
  EXPECT_EQ( p3, p.at( 2 ) );
  EXPECT_EQ( p4, p.at( 3 ) );
}

// ----------------------------------------------------------------------------
TEST(polygon, contains)
{
  polygon p;

  p.push_back( p1 );
  p.push_back( p2 );
  p.push_back( p3 );
  p.push_back( p4 );

  EXPECT_TRUE( p.contains( 30, 30 ) );
  EXPECT_FALSE( p.contains( 70, 70 ) );
}

// ----------------------------------------------------------------------------
TEST(polygon, get_vertices)
{
  polygon p;

  p.push_back( p1 );
  p.push_back( p2 );
  p.push_back( p3 );
  p.push_back( p4 );

  auto vec = p.get_vertices();

  ASSERT_EQ( 4, vec.size() );
  EXPECT_EQ( p1, vec.at( 0 ) );
  EXPECT_EQ( p2, vec.at( 1 ) );
  EXPECT_EQ( p3, vec.at( 2 ) );
  EXPECT_EQ( p4, vec.at( 3 ) );
}
