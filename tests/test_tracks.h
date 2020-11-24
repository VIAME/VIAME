// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 *
 * \brief Various functions for creating a collections of tracks for running tests
 *
 * These functions are shared by various tests
 */

#ifndef KWIVER_TEST_TEST_TRACKS_H_
#define KWIVER_TEST_TEST_TRACKS_H_

#include <random>

#include <vital/types/track_set.h>

namespace kwiver {
namespace testing {

// Generate a set of generic tracks
//
// paramters are:
//   frames - total number of frames to span
//   max_tracks_per_frame - maximum number of track states per frame
//   min_tracks_per_frame - minimum number of track states per frame
//   termination_fraction - fraction of tracks to terminate on each frame
//   skip_fraction - fraction of tracks to miss a state on each frame
//   frame_drop_fraction - fraction of frames with no tracks (skipped)
//
//  if the number of tracks drops below min_tracks_per_frame, create new
//  tracks to achieve max_tracks_per_frame.
kwiver::vital::track_set_sptr
generate_tracks( unsigned frames=100,
                 unsigned max_tracks_per_frame=1000,
                 unsigned min_tracks_per_frame=500,
                 double termination_fraction = 0.1,
                 double skip_fraction = 0.01,
                 double frame_drop_fraction = 0.01 )
{
  using namespace kwiver::vital;

  std::default_random_engine rand_gen(0);
  std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

  track_id_t track_id=0;
  std::vector< track_sptr > all_tracks, active_tracks;
  for( unsigned f=0; f<frames; ++f )
  {
    // randomly decide to skip some frames
    if( uniform_dist(rand_gen) < frame_drop_fraction )
    {
      continue;
    }

    if( active_tracks.size() < min_tracks_per_frame )
    {
      // create tracks as needed to get enough on this frame
      while( active_tracks.size() < max_tracks_per_frame )
      {
        auto t = track::create();
        t->set_id(track_id++);
        active_tracks.push_back(t);
        all_tracks.push_back(t);
      }
    }

    // add a state for each track to this frame
    for( auto t : active_tracks )
    {
      if( t->empty() || uniform_dist(rand_gen) >= skip_fraction )
      {
        t->append( std::make_shared<track_state>( f ) );
      }
    }

    // randomly select tracks to terminate
    std::vector< track_sptr > next_tracks;
    for( auto t : active_tracks )
    {
      if( uniform_dist(rand_gen) >= termination_fraction )
      {
        next_tracks.push_back( t );
      }
    }
    active_tracks.swap( next_tracks );

  }
  return std::make_shared<track_set>( all_tracks );
}

} // end namespace testing
} // end namespace kwiver

#endif
