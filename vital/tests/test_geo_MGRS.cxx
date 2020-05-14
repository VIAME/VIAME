/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
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
 * \brief test geo_MGRS functionality
 */

#include <gtest/gtest.h>

#include <vital/types/geo_MGRS.h>

#include <sstream>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(geo_MGRS, default_constructor)
{
  geo_MGRS gm;
  EXPECT_TRUE(  gm.is_empty() );
  EXPECT_FALSE( gm.is_valid() );
}

// ----------------------------------------------------------------------------
TEST(geo_MGRS, constructor_str)
{
  geo_MGRS gm( "test_str12345" );
  EXPECT_FALSE( gm.is_empty() );
  EXPECT_TRUE(  gm.is_valid() );
}

// ----------------------------------------------------------------------------
TEST(geo_MGRS, coord)
{
  geo_MGRS gm;
  geo_MGRS gm_cpy = gm.set_coord( "test_str12345" );

  EXPECT_EQ( gm.coord(), "test_str12345" );
  EXPECT_EQ( gm_cpy.coord(), "test_str12345" );
  EXPECT_FALSE( gm.is_empty() );
  EXPECT_TRUE(  gm.is_valid() );
  EXPECT_FALSE( gm_cpy.is_empty() );
  EXPECT_TRUE(  gm_cpy.is_valid() );

  gm_cpy = gm.set_coord( "another_test_str" );
  EXPECT_EQ( gm.coord(), "another_test_str" );
  EXPECT_EQ( gm_cpy.coord(), "another_test_str" );
  EXPECT_FALSE( gm.is_empty() );
  EXPECT_TRUE(  gm.is_valid() );
  EXPECT_FALSE( gm_cpy.is_empty() );
  EXPECT_TRUE(  gm_cpy.is_valid() );

  // Test return to empty
  gm_cpy = gm.set_coord( "" );
  EXPECT_EQ( gm.coord(), "" );
  EXPECT_EQ( gm_cpy.coord(), "" );
  EXPECT_TRUE(  gm.is_empty() );
  EXPECT_FALSE( gm.is_valid() );
  EXPECT_TRUE(  gm_cpy.is_empty() );
  EXPECT_FALSE( gm_cpy.is_valid() );
}

// ----------------------------------------------------------------------------
TEST(geo_MGRS, comparisons)
{
  geo_MGRS gm1, gm2;
  EXPECT_TRUE(  gm1 == gm2 );
  EXPECT_FALSE( gm1 != gm2 );

  // Check copies are equal
  geo_MGRS gm1_cpy = gm1.set_coord( "test_str12345" );
  EXPECT_FALSE( gm1 == gm2 );
  EXPECT_TRUE(  gm1 != gm2 );
  EXPECT_TRUE(  gm1_cpy == gm1 );
  EXPECT_FALSE( gm1_cpy != gm1 );
  EXPECT_FALSE( gm1_cpy == gm2 );
  EXPECT_TRUE(  gm1_cpy != gm2 );

  gm2.set_coord( "test_str12345" );
  EXPECT_TRUE(  gm1 == gm2 );
  EXPECT_FALSE( gm1 != gm2 );
  EXPECT_TRUE(  gm1_cpy == gm1 );
  EXPECT_FALSE( gm1_cpy != gm1 );
  EXPECT_TRUE(  gm1_cpy == gm2 );
  EXPECT_FALSE( gm1_cpy != gm2 );
}

// ----------------------------------------------------------------------------
TEST(geo_MGRS, insert_operator_empty)
{
  geo_MGRS gm;
  std::stringstream s;
  s << gm;
  EXPECT_EQ( s.str(), "[MGRS: ]" );
}

// ----------------------------------------------------------------------------
TEST(geo_MGRS, insert_operator)
{
  geo_MGRS gm( "test_str12345" );
  std::stringstream s;
  s << gm;
  EXPECT_EQ( s.str(), "[MGRS: test_str12345]" );
}
