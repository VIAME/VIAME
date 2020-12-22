// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/tests/test_track_set.h>

// ----------------------------------------------------------------------------
int main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(track_set, accessor_functions)
{
  using namespace kwiver::vital::testing;

  auto test_set = make_simple_track_set(1);
  test_track_set_accessors( test_set );
}

// ----------------------------------------------------------------------------
TEST(track_set, modifier_functions)
{
  using namespace kwiver::vital::testing;

  auto test_set = make_simple_track_set(1);
  test_track_set_modifiers( test_set );
}

// ----------------------------------------------------------------------------
TEST(track_set, merge_functions)
{
  using namespace kwiver::vital::testing;

  auto test_set_1 = make_simple_track_set(1);
  auto test_set_2 = make_simple_track_set(2);
  test_track_set_merge(test_set_1, test_set_2);

  auto test_set_3 = std::make_shared< kwiver::vital::track_set >();
  ASSERT_TRUE( test_set_3->empty() );

  test_set_3->merge_in_other_track_set( test_set_2 );

  EXPECT_FALSE( test_set_3->empty() );
  ASSERT_EQ( 4, test_set_3->size() );
}

