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

#include <vital/util/string_editor.h>

#include <gtest/gtest.h>

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(string_editor, test)
{
  kwiver::vital::string_editor se;

  std::string str( "first line\n" );
  std::string astr( str );

  // test empty editor
  se.edit( str );
  EXPECT_EQ( astr, str );

  se.add( new kwiver::vital::edit_operation::shell_comment() );
  se.add( new kwiver::vital::edit_operation::right_trim() );
  se.add( new kwiver::vital::edit_operation::remove_blank_string() );

  EXPECT_TRUE( se.edit(str) );
  EXPECT_EQ( "first line", str );

  str = "  \n";
  EXPECT_FALSE( se.edit( str ) ) << "Blank line should be absorbed";

  str = "trailing spaces        \n";
  EXPECT_TRUE( se.edit(str) );
  EXPECT_EQ( "trailing spaces", str );

  str = "# comment  \n";
  EXPECT_FALSE( se.edit( str ) ) << "Comment should be absorbed";

  str = "foo bar  # trailing comment  \n";
  EXPECT_TRUE( se.edit(str) );
  EXPECT_EQ( "foo bar", str );
}
