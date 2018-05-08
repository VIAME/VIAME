/*ckwg +29
 * Copyright 2014-2017 by Kitware, Inc.
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