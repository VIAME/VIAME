/*ckwg +29
 * Copyright 2014 by Kitware, Inc.
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
#include <vector>

#include <vital/types/track.h>
#include <vital/types/track_set.h>

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

  unsigned track_id = 0;

  std::vector< track_sptr > test_tracks;

  auto test_state1 = std::make_shared<track_state>( 1 );
  auto test_state2 = std::make_shared<track_state>( 2 );
  auto test_state3 = std::make_shared<track_state>( 3 );

  test_tracks.push_back( track::make() ) ;
  test_tracks.back()->append( test_state1 );
  test_tracks.back()->set_id( track_id++ );

  test_tracks.push_back( track::make() ) ;
  test_tracks.back()->append( test_state1->clone() );
  test_tracks.back()->set_id( track_id++ );

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

  track_set_sptr test_set( new simple_track_set( test_tracks ) );

  TEST_EQUAL("Total set size", test_set->size(), 4);

  TEST_EQUAL("Active set size 1", test_set->active_tracks(-1).size(), 3);
  TEST_EQUAL("Active set size 2", test_set->active_tracks(-2).size(), 3);
  TEST_EQUAL("Active set size 3", test_set->active_tracks(-3).size(), 2);

  TEST_EQUAL("Terminated set size", test_set->terminated_tracks(-1).size(), 3);
  TEST_EQUAL("New set size", test_set->new_tracks(-2).size(), 1);

  TEST_EQUAL("Percentage tracked", test_set->percentage_tracked(-1,-2), 0.5);
}
