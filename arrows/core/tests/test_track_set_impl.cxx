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

#include <test_common.h>

#include <iostream>

#include <arrows/core/track_set_impl.h>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}


IMPLEMENT_TEST(accessor_functions)
{
  using namespace kwiver::vital;
  using kwiver::arrows::core::frame_index_track_set_impl;

  unsigned track_id = 0;

  std::vector< track_sptr > test_tracks;

  auto test_state1 = std::make_shared<track_state>( 1 );
  auto test_state2 = std::make_shared<track_state>( 4 );
  auto test_state3 = std::make_shared<track_state>( 9 );

  test_tracks.push_back( track::make() ) ;
  test_tracks.back()->append( test_state1 );
  test_tracks.back()->set_id( track_id++ );

  test_tracks.push_back( track::make() ) ;
  test_tracks.back()->append( test_state1->clone() );
  test_tracks.back()->set_id( track_id++ );

  // skip some track ids
  track_id = 5;

  test_tracks.push_back( track::make() ) ;
  test_tracks.back()->append( test_state2 );
  test_tracks.back()->set_id( track_id++ );

  test_tracks.push_back( track::make() ) ;
  test_tracks.back()->append( test_state3 );
  test_tracks.back()->set_id( track_id++ );

  test_tracks[0]->append( test_state2->clone() );
  test_tracks[0]->append( test_state3->clone() );
  test_tracks[1]->append( test_state2->clone() );
  test_tracks[2]->append( test_state3->clone() );

  std::cout << "tracks sizes: ";
  for(auto t : test_tracks)
  {
    std::cout << t->size() << " ";
  }
  std::cout << std::endl;

  typedef std::unique_ptr<track_set_implementation> tsi_uptr;
  auto test_set = std::make_shared<track_set>(
                    tsi_uptr(new frame_index_track_set_impl(test_tracks) ) );

  TEST_EQUAL("Set empty", test_set->empty(), false);
  TEST_EQUAL("Total set size", test_set->size(), 4);

  TEST_EQUAL("Active set size (-1)", test_set->active_tracks(-1).size(), 3);
  TEST_EQUAL("Active set size (4)", test_set->active_tracks(4).size(), 3);
  TEST_EQUAL("Active set size (1)", test_set->active_tracks(1).size(), 2);
  TEST_EQUAL("Inactive set size (4)", test_set->inactive_tracks(4).size(), 1);
  TEST_EQUAL("Inactive set size (1)", test_set->inactive_tracks(1).size(), 2);

  TEST_EQUAL("Get Track ID (2)", test_set->get_track(2) == nullptr, true);
  TEST_EQUAL("Get Track ID (4)", test_set->get_track(5)->id(), 5);

  std::set<frame_id_t> all_frames = test_set->all_frame_ids();
  std::cout << "frame IDs: ";
  for(auto f : all_frames)
  {
    std::cout << f << " ";
  }
  std::cout << std::endl;
  std::set<frame_id_t> true_frames({1,4,9});
  TEST_EQUAL("Access all frame IDs", std::equal(all_frames.begin(), all_frames.end(),
                                                true_frames.begin()), true);

  std::set<track_id_t> all_track_ids = test_set->all_track_ids();
  std::cout << "track IDs: ";
  for(auto tid : all_track_ids)
  {
    std::cout << tid << " ";
  }
  std::cout << std::endl;
  std::set<track_id_t> true_track_ids({0,1,5,6});
  TEST_EQUAL("Access all track IDs", std::equal(all_track_ids.begin(), all_track_ids.end(),
                                                true_track_ids.begin()), true);

  TEST_EQUAL("First frame ID", test_set->first_frame(), 1);
  TEST_EQUAL("Last frame ID", test_set->last_frame(), 9);
  TEST_EQUAL("Terminated set size", test_set->terminated_tracks(-1).size(), 3);
  TEST_EQUAL("New set size (4)", test_set->new_tracks(4).size(), 1);
  TEST_EQUAL("New set size (-2)", test_set->new_tracks(-2).size(), 0);

  TEST_EQUAL("Percentage tracked", test_set->percentage_tracked(-1,-6), 0.5);
}
