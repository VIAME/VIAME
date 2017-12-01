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
 * \brief test attribute_set functionality
 */

#include <vital/attribute_set.h>

#include <gtest/gtest.h>

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(attribute_set, api)
{
  kwiver::vital::attribute_set set;

  EXPECT_EQ( 0, set.size() );
  EXPECT_TRUE( set.empty() );

  set.add( "first", 1 );
  EXPECT_EQ( 1, set.size() );
  EXPECT_FALSE( set.empty() );

  // Replace entry
  set.add( "first", (double) 3.14159 );
  EXPECT_EQ( 1, set.size() );
  EXPECT_FALSE( set.empty() );
  EXPECT_TRUE( set.has( "first" ) );
  EXPECT_FALSE( set.has( "the-first" ) );

  set.add( "second", 42 );
  EXPECT_EQ( 2, set.size() );
  EXPECT_FALSE( set.empty() );
  EXPECT_TRUE( set.has( "first" ) );
  EXPECT_TRUE( set.has( "second" ) );

  EXPECT_EQ( 42, set.get<int>( "second" ) );
  EXPECT_TRUE( set.is_type<int>( "second") );

  // Test returning wrong type, exception
  EXPECT_THROW(
    set.get<double>( "second" ),
    kwiver::vital::bad_any_cast );

  // Test data() accessor
  EXPECT_THROW(
    set.data( "does-not-exist" ),
    kwiver::vital::attribute_set_exception );

  // Test iterators
  for ( auto it = set.begin(); it != set.end(); ++it )
  {
    std::cout << it->first << std::endl;
  }
}
