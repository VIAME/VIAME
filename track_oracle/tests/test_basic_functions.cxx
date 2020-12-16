// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Test basic track_oracle functionality
 */

#include <gtest/gtest.h>

#include <track_oracle/core/track_oracle_core.h>
#include <track_oracle/core/track_base.h>
#include <track_oracle/data_terms/data_terms.h>

namespace to = ::kwiver::track_oracle;

namespace { //anon

struct demo_track_alpha: public to::track_base< demo_track_alpha >
{
  // alpha tracks have a track_id and a frame_id
  to::track_field< to::dt::tracking::external_id >  track_id;
  to::track_field< to::dt::tracking::frame_number > frame_id;

  // associate them with the track structure
  demo_track_alpha()
  {
    Track.add_field( track_id ); // applies to tracks
    Frame.add_field( frame_id ); // applies to frames
  }
};

struct demo_track_beta: public to::track_base< demo_track_beta >
{
  // beta tracks have the same track_id and frame_id, along with a frame timestamp
  // (field names in the code are different, but the data term is the same)
  to::track_field< to::dt::tracking::external_id >     t_id;
  to::track_field< to::dt::tracking::frame_number >    f_id;
  to::track_field< to::dt::tracking::timestamp_usecs > f_ts;

  demo_track_beta()
  {
    Track.add_field( t_id );
    Frame.add_field( f_id );
    Frame.add_field( f_ts );
  }
};

//
// arbitrary index-to-value functions for testing
//

to::dt::tracking::external_id::Type index2trackID( size_t index )
{
  return (index*17)+13;
}

to::dt::tracking::frame_number::Type index2frameNum( size_t index )
{
  return (index*23)+2;
}

to::dt::tracking::timestamp_usecs::Type index2timestamp( size_t index )
{
  return (index*1000000)+77;
}

} // ...anon

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ------------------------------------------------------------------
TEST(track_oracle, data_term_transfer)
{

  to::track_handle_list_type tracks;
  const size_t n_tracks(5), n_frames(7);
  //
  // create the tracks as type alpha
  //
  {
    demo_track_beta beta;

    for (size_t i=0; i<n_tracks; ++i)
    {
      // create an instance of beta
      to::track_handle_type h = beta.create();
      EXPECT_TRUE( h.is_valid() );
      EXPECT_NE( h.row, to::SYSTEM_ROW_HANDLE );

      // set the track ID
      beta( h ).t_id() = index2trackID( i );
      EXPECT_EQ( beta( h ).t_id(), index2trackID( i ));

      // verify no frames via direct query
      EXPECT_EQ( to::track_oracle_core::get_n_frames( h ), 0 );
      {
        // verify no frames via retrieval
        auto frames = to::track_oracle_core::get_frames( h );
        EXPECT_EQ( frames.size(), 0 );
      }

      // create some frames
      for (size_t j=0; j<n_frames; ++j)
      {
        to::frame_handle_type f = beta( h ).create_frame();
        EXPECT_TRUE( f.is_valid() );
        EXPECT_NE( f.row, to::SYSTEM_ROW_HANDLE );

        // set the frame's ID and timestamp
        beta[ f ].f_id() = index2frameNum( j );
        beta[ f ].f_ts() = index2timestamp( j );
        EXPECT_EQ( beta[ f ].f_id(), index2frameNum( j ));
        EXPECT_EQ( beta[ f ].f_ts(), index2timestamp( j ));
      }

      // verify via direct query
      EXPECT_EQ( to::track_oracle_core::get_n_frames( h ), n_frames );
      {
        // verify via retrieval
        auto frames = to::track_oracle_core::get_frames( h );
        EXPECT_EQ( frames.size(), n_frames );
      }

      // remember the track
      tracks.push_back( h );

    } // ...for each track
  } // ...create the list of beta tracks

  //
  // everything is out of scope except the list of track handles
  //
  EXPECT_EQ( tracks.size(), n_tracks );

  //
  // now verify that we can retrive the data via the handles,
  // both by direct access and via the alpha schema
  //

  {
    demo_track_alpha alpha;
    // need explicit term instance since alpha doesn't have timestamps
    to::track_field< to::dt::tracking::timestamp_usecs > ts;

    for (size_t i=0; i<n_tracks; ++i)
    {
      const auto& h = tracks[i];
      // verify the ID
      EXPECT_EQ( alpha( h ).track_id(), index2trackID( i ));

      // verify the frame data
      auto frames = to::track_oracle_core::get_frames( h );
      EXPECT_EQ( frames.size(), n_frames );
      for (size_t j=0; j<n_frames; ++j)
      {
        const auto& f = frames[ j ];

        // the frame ID is in the alpha schema
        EXPECT_EQ( alpha[ f ].frame_id(), index2frameNum( j ));

        // need to explicitly get the timestamp
        EXPECT_EQ( ts( f.row ), index2timestamp( j ));

      } // ... for all frames
    } // ... for all tracks

  } // ...testing cross-schema transfer
}
