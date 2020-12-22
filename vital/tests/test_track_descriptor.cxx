// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief core track_descriptor class tests
 */

#include <vital/types/track_descriptor.h>

#include <gtest/gtest.h>

// Only run debug tests if in debug mode
#ifndef NDEBUG
#define RUN_DEBUG_TESTS
#endif

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// The following tests are meant only for a debug build,
// as the functions only throw the tested exceptions in debug builds
#ifdef RUN_DEBUG_TESTS
// ----------------------------------------------------------------------------
TEST(track_descriptor, size_with_throw)
{
  track_descriptor_sptr td = track_descriptor::create("foo_type");
  ASSERT_THROW(td->descriptor_size(), std::logic_error);
}

// ----------------------------------------------------------------------------
TEST(track_descriptor, at_with_throw)
{
  track_descriptor_sptr td = track_descriptor::create("foo_type");
  ASSERT_THROW(td->at(0), std::logic_error);
  ASSERT_THROW(td->at(0) = 5, std::logic_error);
}
#endif

// ----------------------------------------------------------------------------
TEST(track_descriptor, has_descriptor)
{
  track_descriptor_sptr td = track_descriptor::create("foo_type");

  // Case where data_ is NULL
  ASSERT_FALSE( td->has_descriptor() );

  td->resize_descriptor( 0 );
  ASSERT_FALSE( td->has_descriptor() );

  td->resize_descriptor( 1 );
  ASSERT_TRUE( td->has_descriptor() );

  td->at(0) = 5;
  ASSERT_TRUE( td->has_descriptor() );
}

// ----------------------------------------------------------------------------
TEST(track_descriptor, size_no_throw)
{
  track_descriptor_sptr td = track_descriptor::create("foo_type");

  td->resize_descriptor( 0 );
  ASSERT_EQ( td->descriptor_size(), 0 );

  td->resize_descriptor( 1 );
  ASSERT_EQ( td->descriptor_size(), 1 );

  td->at(0) = 5;
  ASSERT_EQ( td->descriptor_size(), 1 );

  td->resize_descriptor( 100 );
  ASSERT_EQ( td->descriptor_size(), 100 );

}

// ----------------------------------------------------------------------------
TEST(track_descriptor, at_no_throw)
{
  track_descriptor_sptr td = track_descriptor::create("foo_type");

  td->resize_descriptor( 3 );
  td->at(0) = 5;
  td->at(1) = 10;
  td->at(2) = 15;

  EXPECT_EQ( td->at(0), 5 );
  EXPECT_EQ( td->at(1), 10 );
  EXPECT_EQ( td->at(2), 15 );

  td->resize_descriptor( 1 );
  td->at(0) = 20;

  EXPECT_EQ( td->at(0), 20);
}
