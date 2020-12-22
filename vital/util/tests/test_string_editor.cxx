// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
