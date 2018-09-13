/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * \brief test util string_editor class
 */

#include <vital/util/string.h>

#include <gtest/gtest.h>

#include <sstream>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(string, starts_with)
{
  EXPECT_TRUE( starts_with( "input_string", "input" ) );
  EXPECT_FALSE( starts_with( "input_string", " input" ) );
  EXPECT_TRUE( starts_with( " input_string", " input" ) );
  EXPECT_FALSE( starts_with( "input_string", "string" ) );
}

// ----------------------------------------------------------------------------
TEST(string, format)
{
  EXPECT_EQ( "1 2", string_format( "%d %d", 1, 2 ) );
  EXPECT_EQ( " 1 2", string_format( " %d %d", 1, 2 ) );

  auto const long_string =
    std::string{ "this is a very long string - relatively speaking" };
  EXPECT_EQ( long_string, string_format( "%s", long_string.c_str() ) );
}

// ----------------------------------------------------------------------------
TEST(string, join)
{
  {
    std::vector<std::string> input_vec;
    EXPECT_EQ( "", join( input_vec, ", " ) );
  }

  {
    std::vector<std::string> input_vec{ "one" };
    EXPECT_EQ( "one", join( input_vec, ", " ) );
  }

  {
    std::vector<std::string> input_vec{ "one", "two", "three" };
    EXPECT_EQ( "one, two, three", join( input_vec, ", " ) );
  }

  {
    std::set<std::string> input_set;
    EXPECT_EQ( "", join( input_set, ", " ) );
  }

  {
    std::set<std::string> input_set{ "one" };
    EXPECT_EQ( "one", join( input_set, ", " ) );
  }

  {
    std::set<std::string> input_set{ "one", "three", "two" };
    EXPECT_EQ( "one, three, two", join( input_set, ", " ) );
  }
}
