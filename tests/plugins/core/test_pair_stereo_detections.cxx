/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Unit tests for pair_stereo_detections utility functions and
 *        accumulation logic ported to the core process.
 */

#include <gtest/gtest.h>

#include "pair_stereo_detections.h"

#include <vital/types/bounding_box.h>
#include <vital/types/detected_object.h>
#include <vital/types/object_track_set.h>
#include <vital/types/timestamp.h>
#include <vital/types/camera_intrinsics.h>
#include <vital/types/rotation.h>

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <vector>

namespace kv = kwiver::vital;
using namespace viame::core;

// =============================================================================
// Accumulation support structures (mirrored from process implementation for
// standalone testing)
// =============================================================================

struct IdPair
{
  kv::track_id_t left_id;
  kv::track_id_t right_id;
};

struct Pairing
{
  std::set< kv::frame_id_t > frame_set;
  IdPair left_right_id_pair;
};

struct Range
{
  kv::track_id_t left_id, right_id, new_track_id;
  kv::frame_id_t frame_id_first, frame_id_last;
  int detection_count;
};

static size_t cantor_pairing( size_t i, size_t j )
{
  return ( ( i + j ) * ( i + j + 1u ) ) / 2u + j;
}

// =============================================================================
// Accumulation helper functions (standalone versions for testing)
// =============================================================================

/// Equivalent to priv::create_split_ranges_from_track_pairs()
static std::vector< Range > create_split_ranges(
  const std::map< size_t, Pairing >& pairings,
  int detection_split_threshold,
  kv::track_id_t next_id )
{
  kv::frame_id_t last_frame_id = 0;
  for( const auto& p : pairings )
  {
    if( !p.second.frame_set.empty() )
    {
      kv::frame_id_t last = *p.second.frame_set.rbegin();
      if( last > last_frame_id )
        last_frame_id = last;
    }
  }

  std::map< size_t, std::shared_ptr< Range > > open_ranges;
  std::set< std::shared_ptr< Range > > pending_ranges;
  std::vector< Range > ranges;

  for( kv::frame_id_t i_frame = 0; i_frame <= last_frame_id; i_frame++ )
  {
    for( const auto& pairing : pairings )
    {
      if( pairing.second.frame_set.find( i_frame ) == pairing.second.frame_set.end() )
        continue;

      if( open_ranges.find( pairing.first ) != open_ranges.end() )
      {
        auto& range = open_ranges[pairing.first];
        range->detection_count += 1;
        range->frame_id_last = i_frame + 1;

        if( range->detection_count >= detection_split_threshold )
        {
          std::set< std::shared_ptr< Range > > to_remove;
          for( const auto& pending : pending_ranges )
          {
            if( pending == range )
              continue;
            if( pending->left_id == range->left_id || pending->right_id == range->right_id )
              to_remove.insert( pending );
          }

          if( pending_ranges.find( range ) != pending_ranges.end() )
            pending_ranges.erase( range );

          for( const auto& r : to_remove )
          {
            pending_ranges.erase( r );
            if( r->detection_count >= detection_split_threshold )
            {
              r->frame_id_last = range->frame_id_first - 1;
              ranges.push_back( *r );
            }
          }

          for( auto it = open_ranges.begin(); it != open_ranges.end(); )
          {
            if( to_remove.find( it->second ) != to_remove.end() )
              it = open_ranges.erase( it );
            else
              ++it;
          }
        }
      }
      else
      {
        auto range = std::make_shared< Range >();
        range->left_id = pairing.second.left_right_id_pair.left_id;
        range->right_id = pairing.second.left_right_id_pair.right_id;
        range->new_track_id = next_id++;
        range->detection_count = 1;
        range->frame_id_first = i_frame;
        range->frame_id_last = i_frame + 1;

        open_ranges[pairing.first] = range;
        pending_ranges.insert( range );

        for( auto& op : open_ranges )
        {
          if( op.second == range )
            continue;
          if( op.second->left_id == range->left_id || op.second->right_id == range->right_id )
            pending_ranges.insert( op.second );
        }
      }
    }
  }

  for( auto& op : open_ranges )
  {
    if( op.second->detection_count < detection_split_threshold )
      continue;
    op.second->frame_id_last = std::numeric_limits< int64_t >::max();
    ranges.push_back( *op.second );
  }

  return ranges;
}

/// Equivalent to priv::filter_tracks()
static std::vector< kv::track_sptr > filter_tracks(
  std::vector< kv::track_sptr > tracks,
  int min_track_length, int max_track_length,
  double min_avg_area, double max_avg_area )
{
  if( min_track_length > 0 || max_track_length > 0 )
  {
    int min_len = min_track_length > 0 ? min_track_length : 0;
    int max_len = max_track_length > 0 ? max_track_length : std::numeric_limits< int >::max();

    tracks.erase(
      std::remove_if( tracks.begin(), tracks.end(),
        [min_len, max_len]( const kv::track_sptr& track )
        {
          int sz = static_cast< int >( track->size() );
          return sz < min_len || sz > max_len;
        } ),
      tracks.end() );
  }

  if( min_avg_area > 0.0 || max_avg_area > 0.0 )
  {
    double min_a = min_avg_area > 0.0 ? min_avg_area : 0.0;
    double max_a = max_avg_area > 0.0 ? max_avg_area : std::numeric_limits< double >::max();

    tracks.erase(
      std::remove_if( tracks.begin(), tracks.end(),
        [min_a, max_a]( const kv::track_sptr& track )
        {
          double avg = 0.0;
          int count = 0;
          for( const auto& state : *track | kv::as_object_track )
          {
            if( state->detection() )
            {
              avg += state->detection()->bounding_box().area();
              count++;
            }
          }
          if( count > 0 )
            avg /= static_cast< double >( count );
          return avg < min_a || avg > max_a;
        } ),
      tracks.end() );
  }

  return tracks;
}

// =============================================================================
// Test helpers
// =============================================================================

static kv::detected_object_sptr make_detection( double x1, double y1, double x2, double y2 )
{
  kv::bounding_box_d bbox( x1, y1, x2, y2 );
  return std::make_shared< kv::detected_object >( bbox, 1.0 );
}

static kv::detected_object_sptr make_detection_with_class(
  double x1, double y1, double x2, double y2, const std::string& cls )
{
  kv::bounding_box_d bbox( x1, y1, x2, y2 );
  auto det = std::make_shared< kv::detected_object >( bbox, 1.0 );
  auto dot = std::make_shared< kv::detected_object_type >();
  dot->set_score( cls, 1.0 );
  det->set_type( dot );
  return det;
}

static kv::track_sptr make_track_with_detections(
  kv::track_id_t id,
  const std::vector< std::pair< kv::frame_id_t, kv::bounding_box_d > >& frames )
{
  auto track = kv::track::create();
  track->set_id( id );

  for( const auto& frame : frames )
  {
    auto det = std::make_shared< kv::detected_object >( frame.second, 1.0 );
    kv::timestamp ts( frame.first, frame.first );
    auto state = std::make_shared< kv::object_track_state >( ts, det );
    track->append( state );
  }

  return track;
}

// =============================================================================
// Tests: Cantor Pairing
// =============================================================================

TEST( PairStereoDetectionsTest, cantor_pairing_basic )
{
  // cantor_pairing should produce unique values for different (i,j) pairs
  EXPECT_EQ( cantor_pairing( 0, 0 ), 0u );
  EXPECT_EQ( cantor_pairing( 1, 0 ), 1u );
  EXPECT_EQ( cantor_pairing( 0, 1 ), 2u );
  EXPECT_EQ( cantor_pairing( 1, 1 ), 4u );
  EXPECT_EQ( cantor_pairing( 2, 0 ), 3u );
}

TEST( PairStereoDetectionsTest, cantor_pairing_uniqueness )
{
  // All pairs (i,j) for i,j in [0,10) should produce unique keys
  std::set< size_t > keys;
  for( size_t i = 0; i < 10; ++i )
  {
    for( size_t j = 0; j < 10; ++j )
    {
      keys.insert( cantor_pairing( i, j ) );
    }
  }
  EXPECT_EQ( keys.size(), 100u );
}

TEST( PairStereoDetectionsTest, cantor_pairing_not_symmetric )
{
  // cantor_pairing(a,b) != cantor_pairing(b,a) for a != b
  EXPECT_NE( cantor_pairing( 1, 2 ), cantor_pairing( 2, 1 ) );
  EXPECT_NE( cantor_pairing( 3, 7 ), cantor_pairing( 7, 3 ) );
}

// =============================================================================
// Tests: IOU Matching Options
// =============================================================================

TEST( PairStereoDetectionsTest, iou_options_defaults )
{
  iou_matching_options opts;
  EXPECT_DOUBLE_EQ( opts.iou_threshold, 0.1 );
  EXPECT_TRUE( opts.require_class_match );
  EXPECT_TRUE( opts.use_optimal_assignment );
}

// =============================================================================
// Tests: Epipolar IOU Matching Options
// =============================================================================

TEST( PairStereoDetectionsTest, epipolar_iou_options_defaults )
{
  epipolar_iou_matching_options opts;
  EXPECT_DOUBLE_EQ( opts.iou_threshold, 0.1 );
  EXPECT_DOUBLE_EQ( opts.default_depth, 5.0 );
  EXPECT_TRUE( opts.require_class_match );
  EXPECT_TRUE( opts.use_optimal_assignment );
}

// =============================================================================
// Tests: Keypoint Projection Matching Options
// =============================================================================

TEST( PairStereoDetectionsTest, keypoint_projection_options_defaults )
{
  keypoint_projection_matching_options opts;
  EXPECT_DOUBLE_EQ( opts.max_keypoint_distance, 50.0 );
  EXPECT_DOUBLE_EQ( opts.default_depth, 5.0 );
  EXPECT_TRUE( opts.require_class_match );
  EXPECT_TRUE( opts.use_optimal_assignment );
}

// =============================================================================
// Tests: IOU Matching
// =============================================================================

TEST( PairStereoDetectionsTest, iou_matching_overlapping_detections )
{
  std::vector< kv::detected_object_sptr > dets1 = {
    make_detection( 0, 0, 100, 100 ),
    make_detection( 200, 200, 300, 300 )
  };

  std::vector< kv::detected_object_sptr > dets2 = {
    make_detection( 10, 10, 110, 110 ),    // overlaps with dets1[0]
    make_detection( 500, 500, 600, 600 )   // no overlap
  };

  iou_matching_options opts;
  opts.iou_threshold = 0.1;
  opts.require_class_match = false;

  auto matches = find_stereo_matches_iou( dets1, dets2, opts );

  ASSERT_EQ( matches.size(), 1u );
  EXPECT_EQ( matches[0].first, 0 );
  EXPECT_EQ( matches[0].second, 0 );
}

TEST( PairStereoDetectionsTest, iou_matching_no_overlap )
{
  std::vector< kv::detected_object_sptr > dets1 = {
    make_detection( 0, 0, 50, 50 )
  };

  std::vector< kv::detected_object_sptr > dets2 = {
    make_detection( 200, 200, 300, 300 )
  };

  iou_matching_options opts;
  opts.iou_threshold = 0.1;
  opts.require_class_match = false;

  auto matches = find_stereo_matches_iou( dets1, dets2, opts );

  EXPECT_EQ( matches.size(), 0u );
}

TEST( PairStereoDetectionsTest, iou_matching_with_class_match )
{
  std::vector< kv::detected_object_sptr > dets1 = {
    make_detection_with_class( 0, 0, 100, 100, "fish" ),
    make_detection_with_class( 200, 200, 300, 300, "crab" )
  };

  std::vector< kv::detected_object_sptr > dets2 = {
    make_detection_with_class( 10, 10, 110, 110, "crab" ),    // overlaps dets1[0] but wrong class
    make_detection_with_class( 210, 210, 310, 310, "crab" )   // overlaps dets1[1] and matches class
  };

  iou_matching_options opts;
  opts.iou_threshold = 0.1;
  opts.require_class_match = true;

  auto matches = find_stereo_matches_iou( dets1, dets2, opts );

  ASSERT_EQ( matches.size(), 1u );
  EXPECT_EQ( matches[0].first, 1 );   // crab from left
  EXPECT_EQ( matches[0].second, 1 );  // crab from right
}

TEST( PairStereoDetectionsTest, iou_matching_empty_inputs )
{
  std::vector< kv::detected_object_sptr > empty;
  std::vector< kv::detected_object_sptr > dets = {
    make_detection( 0, 0, 100, 100 )
  };

  iou_matching_options opts;
  opts.require_class_match = false;

  EXPECT_EQ( find_stereo_matches_iou( empty, dets, opts ).size(), 0u );
  EXPECT_EQ( find_stereo_matches_iou( dets, empty, opts ).size(), 0u );
  EXPECT_EQ( find_stereo_matches_iou( empty, empty, opts ).size(), 0u );
}

TEST( PairStereoDetectionsTest, iou_matching_multiple_matches )
{
  // Three overlapping pairs
  std::vector< kv::detected_object_sptr > dets1 = {
    make_detection( 0, 0, 100, 100 ),
    make_detection( 200, 0, 300, 100 ),
    make_detection( 400, 0, 500, 100 )
  };

  std::vector< kv::detected_object_sptr > dets2 = {
    make_detection( 5, 5, 105, 105 ),
    make_detection( 205, 5, 305, 105 ),
    make_detection( 405, 5, 505, 105 )
  };

  iou_matching_options opts;
  opts.iou_threshold = 0.1;
  opts.require_class_match = false;

  auto matches = find_stereo_matches_iou( dets1, dets2, opts );

  EXPECT_EQ( matches.size(), 3u );
}

// =============================================================================
// Tests: Track Filter
// =============================================================================

TEST( PairStereoDetectionsTest, filter_tracks_by_length )
{
  kv::bounding_box_d bbox( 0, 0, 100, 100 );
  std::vector< kv::track_sptr > tracks = {
    make_track_with_detections( 1, { { 0, bbox } } ),                               // 1 det
    make_track_with_detections( 2, { { 0, bbox }, { 1, bbox }, { 2, bbox } } ),      // 3 dets
    make_track_with_detections( 3, { { 0, bbox }, { 1, bbox }, { 2, bbox },
                                     { 3, bbox }, { 4, bbox } } ),                   // 5 dets
  };

  // Filter: min=2, max=4
  auto filtered = filter_tracks( tracks, 2, 4, 0.0, 0.0 );

  ASSERT_EQ( filtered.size(), 1u );
  EXPECT_EQ( filtered[0]->id(), 2 );
}

TEST( PairStereoDetectionsTest, filter_tracks_by_area )
{
  std::vector< kv::track_sptr > tracks = {
    make_track_with_detections( 1, { { 0, kv::bounding_box_d( 0, 0, 5, 5 ) } } ),       // area=25
    make_track_with_detections( 2, { { 0, kv::bounding_box_d( 0, 0, 10, 10 ) } } ),     // area=100
    make_track_with_detections( 3, { { 0, kv::bounding_box_d( 0, 0, 100, 100 ) } } ),   // area=10000
  };

  // Filter: min_area=50, max_area=500
  auto filtered = filter_tracks( tracks, 0, 0, 50.0, 500.0 );

  ASSERT_EQ( filtered.size(), 1u );
  EXPECT_EQ( filtered[0]->id(), 2 );
}

TEST( PairStereoDetectionsTest, filter_tracks_no_filter )
{
  kv::bounding_box_d bbox( 0, 0, 100, 100 );
  std::vector< kv::track_sptr > tracks = {
    make_track_with_detections( 1, { { 0, bbox } } ),
    make_track_with_detections( 2, { { 0, bbox }, { 1, bbox } } ),
  };

  // No filtering (all zeros)
  auto filtered = filter_tracks( tracks, 0, 0, 0.0, 0.0 );

  EXPECT_EQ( filtered.size(), 2u );
}

TEST( PairStereoDetectionsTest, filter_tracks_min_length_only )
{
  kv::bounding_box_d bbox( 0, 0, 100, 100 );
  std::vector< kv::track_sptr > tracks = {
    make_track_with_detections( 1, { { 0, bbox } } ),
    make_track_with_detections( 2, { { 0, bbox }, { 1, bbox }, { 2, bbox } } ),
  };

  // Min length = 2, no max
  auto filtered = filter_tracks( tracks, 2, 0, 0.0, 0.0 );

  ASSERT_EQ( filtered.size(), 1u );
  EXPECT_EQ( filtered[0]->id(), 2 );
}

// =============================================================================
// Tests: Split Ranges from Track Pairs
// =============================================================================

TEST( PairStereoDetectionsTest, split_continuous_pair_yields_one_range )
{
  size_t key = cantor_pairing( 1, 3 );
  std::map< size_t, Pairing > pairings = {
    { key, { { 1, 2, 3, 8 }, { 1, 3 } } }
  };

  auto ranges = create_split_ranges( pairings, 3, 100 );

  ASSERT_EQ( ranges.size(), 1u );
  EXPECT_EQ( ranges[0].detection_count, 4 );
  EXPECT_EQ( ranges[0].left_id, 1 );
  EXPECT_EQ( ranges[0].right_id, 3 );
  EXPECT_EQ( ranges[0].frame_id_first, 1 );
  EXPECT_GT( ranges[0].frame_id_last, 8 );
}

TEST( PairStereoDetectionsTest, split_inconclusive_pairs_yields_no_result )
{
  std::map< size_t, Pairing > pairings = {
    { cantor_pairing( 1, 3 ), { { 1 }, { 1, 3 } } },
    { cantor_pairing( 1, 4 ), { { 2 }, { 1, 4 } } },
    { cantor_pairing( 1, 5 ), { { 3 }, { 1, 5 } } }
  };

  auto ranges = create_split_ranges( pairings, 3, 100 );

  EXPECT_EQ( ranges.size(), 0u );
}

TEST( PairStereoDetectionsTest, split_separate_pairs_keeps_both )
{
  std::map< size_t, Pairing > pairings = {
    { cantor_pairing( 1, 3 ), { { 1, 2, 3, 4, 5 }, { 1, 3 } } },
    { cantor_pairing( 2, 6 ), { { 1, 2, 3, 4, 5 }, { 2, 6 } } }
  };

  auto ranges = create_split_ranges( pairings, 3, 100 );

  ASSERT_EQ( ranges.size(), 2u );
  EXPECT_EQ( ranges[0].left_id, 1 );
  EXPECT_EQ( ranges[0].right_id, 3 );
  EXPECT_EQ( ranges[1].left_id, 2 );
  EXPECT_EQ( ranges[1].right_id, 6 );
}

TEST( PairStereoDetectionsTest, split_conflicting_pairs_creates_multiple_ranges )
{
  std::map< size_t, Pairing > pairings = {
    { cantor_pairing( 1, 3 ), { { 1, 2, 3, 7, 8, 9 }, { 1, 3 } } },
    { cantor_pairing( 1, 6 ), { { 4, 5, 6 },           { 1, 6 } } }
  };

  auto ranges = create_split_ranges( pairings, 3, 100 );

  ASSERT_EQ( ranges.size(), 3u );

  // First segment: left=1, right=3, frames 1-3
  EXPECT_EQ( ranges[0].left_id, 1 );
  EXPECT_EQ( ranges[0].right_id, 3 );
  EXPECT_EQ( ranges[0].frame_id_last, 3 );

  // Second segment: left=1, right=6, frames 4-6
  EXPECT_EQ( ranges[1].left_id, 1 );
  EXPECT_EQ( ranges[1].right_id, 6 );
  EXPECT_EQ( ranges[1].frame_id_last, 6 );

  // Third segment: left=1, right=3, frames 7+
  EXPECT_EQ( ranges[2].left_id, 1 );
  EXPECT_EQ( ranges[2].right_id, 3 );
  EXPECT_GT( ranges[2].frame_id_last, 9 );
}

TEST( PairStereoDetectionsTest, split_interleaved_inconclusive_yields_one_range )
{
  std::map< size_t, Pairing > pairings = {
    { cantor_pairing( 1, 3 ), { { 1, 2, 3, 5, 7, 9 }, { 1, 3 } } },
    { cantor_pairing( 1, 6 ), { { 4 },                  { 1, 6 } } },
    { cantor_pairing( 1, 7 ), { { 6 },                  { 1, 7 } } }
  };

  auto ranges = create_split_ranges( pairings, 3, 100 );

  ASSERT_EQ( ranges.size(), 1u );
  EXPECT_EQ( ranges[0].left_id, 1 );
  EXPECT_EQ( ranges[0].right_id, 3 );
  EXPECT_GT( ranges[0].frame_id_last, 9 );
}

// =============================================================================
// Tests: Most Likely Pairing (logic only)
// =============================================================================

TEST( PairStereoDetectionsTest, most_likely_picks_highest_frame_count )
{
  // Simulate accumulated pairings
  std::map< size_t, Pairing > pairings = {
    { cantor_pairing( 1, 10 ), { { 1, 2 },          { 1, 10 } } },  // 2 frames
    { cantor_pairing( 1, 11 ), { { 3, 4, 5, 6, 7 }, { 1, 11 } } },  // 5 frames
    { cantor_pairing( 2, 12 ), { { 1, 2, 3 },       { 2, 12 } } },  // 3 frames
  };

  // Find most likely pair for each left track
  struct MostLikelyPair
  {
    int frame_count = -1;
    kv::track_id_t right_id = -1;
  };

  std::map< kv::track_id_t, MostLikelyPair > most_likely;

  for( const auto& pair : pairings )
  {
    kv::track_id_t left_id = pair.second.left_right_id_pair.left_id;
    int count = static_cast< int >( pair.second.frame_set.size() );

    if( most_likely.find( left_id ) == most_likely.end() )
      most_likely[left_id] = MostLikelyPair{};

    if( count > most_likely[left_id].frame_count )
    {
      most_likely[left_id].frame_count = count;
      most_likely[left_id].right_id = pair.second.left_right_id_pair.right_id;
    }
  }

  EXPECT_EQ( most_likely[1].right_id, 11 );  // 5 frames beats 2 frames
  EXPECT_EQ( most_likely[1].frame_count, 5 );
  EXPECT_EQ( most_likely[2].right_id, 12 );
  EXPECT_EQ( most_likely[2].frame_count, 3 );
}

// =============================================================================
// Tests: Accumulate Frame Pairings (logic)
// =============================================================================

TEST( PairStereoDetectionsTest, accumulate_frame_pairings_builds_tracks )
{
  // Simulate what accumulate_frame_pairings does
  std::map< kv::track_id_t, kv::track_sptr > acc_tracks1, acc_tracks2;
  std::map< size_t, Pairing > acc_pairings;

  auto store_frame = [&](
    const std::vector< std::pair< int, int > >& matches,
    const std::vector< kv::track_id_t >& tids1,
    const std::vector< kv::track_id_t >& tids2,
    const std::vector< kv::detected_object_sptr >& dets1,
    const std::vector< kv::detected_object_sptr >& dets2,
    kv::frame_id_t frame )
  {
    kv::timestamp ts( frame, frame );

    for( size_t i = 0; i < dets1.size(); ++i )
    {
      kv::track_id_t tid = tids1[i];
      if( acc_tracks1.find( tid ) == acc_tracks1.end() )
      {
        acc_tracks1[tid] = kv::track::create();
        acc_tracks1[tid]->set_id( tid );
      }
      auto state = std::make_shared< kv::object_track_state >( ts, dets1[i] );
      acc_tracks1[tid]->append( state );
    }

    for( size_t i = 0; i < dets2.size(); ++i )
    {
      kv::track_id_t tid = tids2[i];
      if( acc_tracks2.find( tid ) == acc_tracks2.end() )
      {
        acc_tracks2[tid] = kv::track::create();
        acc_tracks2[tid]->set_id( tid );
      }
      auto state = std::make_shared< kv::object_track_state >( ts, dets2[i] );
      acc_tracks2[tid]->append( state );
    }

    for( const auto& match : matches )
    {
      kv::track_id_t left_id = tids1[match.first];
      kv::track_id_t right_id = tids2[match.second];
      size_t key = cantor_pairing( static_cast< size_t >( left_id ),
                                   static_cast< size_t >( right_id ) );

      if( acc_pairings.find( key ) == acc_pairings.end() )
        acc_pairings[key] = Pairing{ {}, { left_id, right_id } };

      acc_pairings[key].frame_set.insert( frame );
    }
  };

  // Frame 0: tracks 1,2 on left, tracks 10,11 on right; match (0->0), (1->1)
  std::vector< kv::track_id_t > tids1 = { 1, 2 };
  std::vector< kv::track_id_t > tids2 = { 10, 11 };
  std::vector< kv::detected_object_sptr > dets1 = {
    make_detection( 0, 0, 100, 100 ),
    make_detection( 200, 0, 300, 100 )
  };
  std::vector< kv::detected_object_sptr > dets2 = {
    make_detection( 5, 5, 105, 105 ),
    make_detection( 205, 5, 305, 105 )
  };
  std::vector< std::pair< int, int > > matches = { { 0, 0 }, { 1, 1 } };

  store_frame( matches, tids1, tids2, dets1, dets2, 0 );

  // Frame 1: same tracks, same matches
  store_frame( matches, tids1, tids2, dets1, dets2, 1 );

  // Verify accumulated state
  EXPECT_EQ( acc_tracks1.size(), 2u );
  EXPECT_EQ( acc_tracks2.size(), 2u );
  EXPECT_EQ( acc_tracks1[1]->size(), 2u );  // 2 frames
  EXPECT_EQ( acc_tracks1[2]->size(), 2u );
  EXPECT_EQ( acc_pairings.size(), 2u );

  size_t key1 = cantor_pairing( 1, 10 );
  EXPECT_EQ( acc_pairings[key1].frame_set.size(), 2u );
  EXPECT_EQ( acc_pairings[key1].left_right_id_pair.left_id, 1 );
  EXPECT_EQ( acc_pairings[key1].left_right_id_pair.right_id, 10 );
}

// =============================================================================
// Tests: Feature Matching Algorithms struct
// =============================================================================

TEST( PairStereoDetectionsTest, feature_matching_algorithms_validity )
{
  feature_matching_algorithms algos;

  // Empty by default — not valid
  EXPECT_FALSE( algos.is_valid() );
  EXPECT_FALSE( algos.has_homography_estimator() );
}

// =============================================================================
// Tests: Calibration Matching Options
// =============================================================================

TEST( PairStereoDetectionsTest, calibration_options_defaults )
{
  calibration_matching_options opts;
  EXPECT_DOUBLE_EQ( opts.max_reprojection_error, 10.0 );
  EXPECT_DOUBLE_EQ( opts.default_depth, 5.0 );
  EXPECT_TRUE( opts.require_class_match );
  EXPECT_TRUE( opts.use_optimal_assignment );
}

// =============================================================================
// Tests: Feature Matching Options
// =============================================================================

TEST( PairStereoDetectionsTest, feature_matching_options_defaults )
{
  feature_matching_options opts;
  EXPECT_EQ( opts.min_feature_match_count, 5 );
  EXPECT_DOUBLE_EQ( opts.min_feature_match_ratio, 0.1 );
  EXPECT_TRUE( opts.use_homography_filtering );
  EXPECT_DOUBLE_EQ( opts.homography_inlier_threshold, 5.0 );
  EXPECT_DOUBLE_EQ( opts.min_homography_inlier_ratio, 0.5 );
  EXPECT_DOUBLE_EQ( opts.box_expansion_factor, 1.1 );
  EXPECT_TRUE( opts.require_class_match );
  EXPECT_TRUE( opts.use_optimal_assignment );
}

// =============================================================================
// Tests: IOU threshold sensitivity
// =============================================================================

TEST( PairStereoDetectionsTest, iou_matching_threshold_sensitivity )
{
  // Two boxes with partial overlap
  std::vector< kv::detected_object_sptr > dets1 = {
    make_detection( 0, 0, 100, 100 )
  };

  // This box overlaps by ~25% with dets1[0]
  std::vector< kv::detected_object_sptr > dets2 = {
    make_detection( 50, 50, 150, 150 )
  };

  iou_matching_options opts;
  opts.require_class_match = false;

  // With low threshold, should match
  opts.iou_threshold = 0.1;
  auto matches_low = find_stereo_matches_iou( dets1, dets2, opts );
  EXPECT_EQ( matches_low.size(), 1u );

  // With high threshold, should not match
  opts.iou_threshold = 0.5;
  auto matches_high = find_stereo_matches_iou( dets1, dets2, opts );
  EXPECT_EQ( matches_high.size(), 0u );
}

// =============================================================================
// Tests: Track creation and splitting helpers
// =============================================================================

TEST( PairStereoDetectionsTest, make_track_with_detections_helper )
{
  kv::bounding_box_d bbox( 10, 20, 30, 40 );
  auto track = make_track_with_detections( 42, { { 0, bbox }, { 1, bbox }, { 2, bbox } } );

  EXPECT_EQ( track->id(), 42 );
  EXPECT_EQ( track->size(), 3u );
  EXPECT_EQ( track->first_frame(), 0 );
  EXPECT_EQ( track->last_frame(), 2 );

  // Verify detection in first state
  auto state = std::dynamic_pointer_cast< kv::object_track_state >( track->front() );
  ASSERT_TRUE( state != nullptr );
  ASSERT_TRUE( state->detection() != nullptr );
  EXPECT_DOUBLE_EQ( state->detection()->bounding_box().upper_left().x(), 10 );
}

TEST( PairStereoDetectionsTest, split_threshold_filters_short_segments )
{
  // Create a pairing that only appears in 2 frames (below threshold of 3)
  std::map< size_t, Pairing > pairings = {
    { cantor_pairing( 1, 3 ), { { 1, 2 }, { 1, 3 } } }  // only 2 frames
  };

  auto ranges = create_split_ranges( pairings, 3, 100 );

  // Should produce no ranges since 2 < threshold of 3
  EXPECT_EQ( ranges.size(), 0u );
}

TEST( PairStereoDetectionsTest, split_threshold_keeps_long_segments )
{
  // Create a pairing that appears in 5 frames (above threshold of 3)
  std::map< size_t, Pairing > pairings = {
    { cantor_pairing( 1, 3 ), { { 1, 2, 3, 4, 5 }, { 1, 3 } } }
  };

  auto ranges = create_split_ranges( pairings, 3, 100 );

  ASSERT_EQ( ranges.size(), 1u );
  EXPECT_EQ( ranges[0].detection_count, 5 );
}

// =============================================================================
// Tests: Combined length and area filtering
// =============================================================================

TEST( PairStereoDetectionsTest, filter_tracks_combined_filters )
{
  std::vector< kv::track_sptr > tracks = {
    // Track 1: 1 det, area=100 → filtered by length
    make_track_with_detections( 1, { { 0, kv::bounding_box_d( 0, 0, 10, 10 ) } } ),
    // Track 2: 3 dets, area=100 → passes both
    make_track_with_detections( 2, { { 0, kv::bounding_box_d( 0, 0, 10, 10 ) },
                                     { 1, kv::bounding_box_d( 0, 0, 10, 10 ) },
                                     { 2, kv::bounding_box_d( 0, 0, 10, 10 ) } } ),
    // Track 3: 3 dets, area=10000 → filtered by area
    make_track_with_detections( 3, { { 0, kv::bounding_box_d( 0, 0, 100, 100 ) },
                                     { 1, kv::bounding_box_d( 0, 0, 100, 100 ) },
                                     { 2, kv::bounding_box_d( 0, 0, 100, 100 ) } } ),
  };

  // min_length=2, max_area=500
  auto filtered = filter_tracks( tracks, 2, 0, 0.0, 500.0 );

  ASSERT_EQ( filtered.size(), 1u );
  EXPECT_EQ( filtered[0]->id(), 2 );
}
