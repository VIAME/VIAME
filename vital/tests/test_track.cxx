// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test core track class
 */

#include <vital/types/track.h>

#include <gtest/gtest.h>

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(track, id)
{
  auto t = kwiver::vital::track::create();
  EXPECT_EQ( kwiver::vital::invalid_track_id, t->id() );

  t->set_id( 25 );
  EXPECT_EQ( 25, t->id() );
}

// ----------------------------------------------------------------------------
TEST(track, insert_remove)
{
  auto t = kwiver::vital::track::create();
  auto ts1 = std::make_shared < kwiver::vital::track_state>(1);
  auto ts2 = std::make_shared < kwiver::vital::track_state>(2);
  auto ts4 = std::make_shared < kwiver::vital::track_state>(4);
  auto ts7 = std::make_shared < kwiver::vital::track_state>(7);
  auto ts8 = std::make_shared < kwiver::vital::track_state>(8);

  t->insert(ts1);
  t->insert(ts2);
  t->insert(ts4);
  t->insert(ts8);

  EXPECT_EQ(4, t->size());
  EXPECT_TRUE(t->remove(ts2));
  EXPECT_EQ(3, t->size());
  EXPECT_FALSE(t->remove(ts2));
  EXPECT_EQ(3, t->size());
  EXPECT_TRUE(t->remove(ts1));
  EXPECT_EQ(2, t->size());
  EXPECT_TRUE(t->remove(ts4));
  EXPECT_EQ(1, t->size());
  EXPECT_TRUE(t->remove(ts8));
  EXPECT_EQ(0, t->size());
  EXPECT_FALSE(t->remove(ts1));
  t->insert(ts7);
  EXPECT_EQ(1, t->size());
  EXPECT_TRUE(t->remove(ts7));
  EXPECT_FALSE(t->remove(ts7));
}

// ----------------------------------------------------------------------------
TEST(track, remove_by_frame)
{
  auto t = kwiver::vital::track::create();
  auto ts1 = std::make_shared< kwiver::vital::track_state >( 1 );
  auto ts2 = std::make_shared< kwiver::vital::track_state >( 2 );
  auto ts4 = std::make_shared< kwiver::vital::track_state >( 4 );

  t->insert( ts1 );
  t->insert( ts2 );
  t->insert( ts4 );

  EXPECT_EQ( 3, t->size() );
  EXPECT_TRUE( t->remove( 2 ) );
  EXPECT_EQ( 2, t->size() );
  EXPECT_FALSE( t->remove( 2 ) );
  EXPECT_EQ( 2, t->size() );

  EXPECT_EQ( t, ts1->track() );
  EXPECT_EQ( t, ts4->track() );
  EXPECT_EQ( nullptr, ts2->track() );
}

// ----------------------------------------------------------------------------
TEST(track, contains)
{
  auto t = kwiver::vital::track::create();
  auto ts1 = std::make_shared< kwiver::vital::track_state >( 1 );
  auto ts4 = std::make_shared< kwiver::vital::track_state >( 4 );

  t->insert( ts1 );
  t->insert( ts4 );

  EXPECT_EQ( 2, t->size() );
  EXPECT_TRUE( t->contains( 1 ) );
  EXPECT_TRUE( t->contains( 4 ) );
  EXPECT_FALSE( t->contains( 2 ) );
  EXPECT_FALSE( t->contains( 5 ) );
}
