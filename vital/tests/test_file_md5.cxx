// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test file md5 computation
 */

#include <tests/test_gtest.h>

#include <vital/vital_types.h>
#include <vital/util/file_md5.h>

#include <string>

using std::string;

kwiver::vital::path_t g_test_file;
string g_ref_value;

// ----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest( &argc, argv );
  GET_ARG(1, g_test_file);
  GET_ARG(2, g_ref_value);
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(file_md5, file)
{
  string test_md5 = kwiver::vital::file_md5( g_test_file );
  EXPECT_EQ( test_md5, g_ref_value );
}
