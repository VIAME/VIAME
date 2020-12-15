// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test util string_editor class
 */

#include <vital/util/data_stream_reader.h>

#include <gtest/gtest.h>

#include <sstream>

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(data_stream_reader,  test)
{
  std::stringstream str;

  str << "first line\n"
         "second line\n"
         "  \n" // blank line
         "# comment\n"
         "foo bar # trailing comment\n";

  kwiver::vital::data_stream_reader dsr{ str };
  std::string line;

  EXPECT_TRUE( dsr.getline( line ) );
  EXPECT_EQ( "first line", line );
  EXPECT_EQ( 1, dsr.line_number() );

  EXPECT_TRUE( dsr.getline( line ) );
  EXPECT_EQ( "second line", line );
  EXPECT_EQ( 2, dsr.line_number() );

  EXPECT_TRUE( dsr.getline( line ) );
  EXPECT_EQ( "foo bar", line );
  EXPECT_EQ( 5, dsr.line_number() );
}
