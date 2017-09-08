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

#include <algorithm>
#include <iostream>

#include <arrows/core/track_set_impl.h>
#include <test_tracks.h>
#include <vital/tests/test_track_set.h>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}


IMPLEMENT_TEST(frame_index_accessor_functions)
{
  using namespace kwiver::vital;
  using namespace kwiver::vital::testing;
  using kwiver::arrows::core::frame_index_track_set_impl;

  auto test_set = make_simple_track_set();

  typedef std::unique_ptr<track_set_implementation> tsi_uptr;
  test_set = std::make_shared<track_set>(
               tsi_uptr(new frame_index_track_set_impl(test_set->tracks() ) ) );

  test_track_set_accessors(test_set);
}


IMPLEMENT_TEST(frame_index_modifier_functions)
{
  using namespace kwiver::vital;
  using namespace kwiver::vital::testing;
  using kwiver::arrows::core::frame_index_track_set_impl;

  auto test_set = make_simple_track_set();

  typedef std::unique_ptr<track_set_implementation> tsi_uptr;
  test_set = std::make_shared<track_set>(
               tsi_uptr(new frame_index_track_set_impl(test_set->tracks() ) ) );

  test_track_set_modifiers(test_set);
}


IMPLEMENT_TEST(frame_index_matches_simple)
{
  using namespace kwiver::vital;
  using kwiver::arrows::core::frame_index_track_set_impl;

  auto tracks = kwiver::testing::generate_tracks();

  typedef std::unique_ptr<track_set_implementation> tsi_uptr;
  auto ftracks = std::make_shared<track_set>(
                   tsi_uptr(new frame_index_track_set_impl(tracks->tracks()) ) );

  TEST_EQUAL("frame_index size() matches simple",
             tracks->size(), ftracks->size());
  TEST_EQUAL("frame_index empty() matches simple",
             tracks->empty(), ftracks->empty());
  TEST_EQUAL("frame_index first_frame() matches simple",
             tracks->first_frame(), ftracks->first_frame());
  TEST_EQUAL("frame_index last_frame() matches simple",
             tracks->last_frame(), ftracks->last_frame());

  auto all_frames_s = tracks->all_frame_ids();
  auto all_frames_f = ftracks->all_frame_ids();
  TEST_EQUAL("frame_index all_frame_ids() matches simple",
             std::equal(all_frames_s.begin(), all_frames_s.end(),
                        all_frames_f.begin()), true);

  auto all_tid_s = tracks->all_track_ids();
  auto all_tid_f = ftracks->all_track_ids();
  TEST_EQUAL("frame_index all_track_ids() matches simple",
             std::equal(all_tid_s.begin(), all_tid_s.end(),
                        all_tid_f.begin()), true);

  auto active_s = tracks->active_tracks(5);
  auto active_f = ftracks->active_tracks(5);
  std::sort(active_s.begin(), active_s.end());
  std::sort(active_f.begin(), active_f.end());
  TEST_EQUAL("frame_index active_tracks() matches simple",
             std::equal(active_s.begin(), active_s.end(),
                        active_f.begin()), true);

  auto inactive_s = tracks->inactive_tracks(15);
  auto inactive_f = ftracks->inactive_tracks(15);
  std::sort(inactive_s.begin(), inactive_s.end());
  std::sort(inactive_f.begin(), inactive_f.end());
  TEST_EQUAL("frame_index inactive_tracks() matches simple",
             std::equal(inactive_s.begin(), inactive_s.end(),
                        inactive_f.begin()), true);

  auto new_s = tracks->new_tracks(40);
  auto new_f = ftracks->new_tracks(40);
  std::sort(new_s.begin(), new_s.end());
  std::sort(new_f.begin(), new_f.end());
  TEST_EQUAL("frame_index new_tracks() matches simple",
             std::equal(new_s.begin(), new_s.end(),
                        new_f.begin()), true);

  auto term_s = tracks->terminated_tracks(60);
  auto term_f = ftracks->terminated_tracks(60);
  std::sort(term_s.begin(), term_s.end());
  std::sort(term_f.begin(), term_f.end());
  TEST_EQUAL("frame_index terminated_tracks() matches simple",
             std::equal(term_s.begin(), term_s.end(),
                        term_f.begin()), true);

  TEST_EQUAL("frame_index percentage_tracked() matches simple",
             tracks->percentage_tracked(10, 50),
             ftracks->percentage_tracked(10, 50));
}
