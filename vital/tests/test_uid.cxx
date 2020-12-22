// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test uuid functionality
 */

#include <vital/types/uid.h>

#include <gtest/gtest.h>

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(uid, api)
{
  kwiver::vital::uid foo{ "test0123456789" };
  EXPECT_TRUE( foo.is_valid() );
  EXPECT_EQ( "test0123456789", std::string{ foo.value() } );

  kwiver::vital::uid foo_copied = foo;
  EXPECT_EQ( foo, foo_copied );

  kwiver::vital::uid foo_assigned;
  foo_assigned = foo;
  EXPECT_EQ( foo, foo_assigned );
}
