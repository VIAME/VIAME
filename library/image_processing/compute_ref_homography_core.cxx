// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of compute_ref_homography_core

#include "compute_ref_homography_core.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <vector>

#include <Eigen/LU>

using namespace kwiver::vital;

namespace kwiver {

namespace arrows {

namespace core {

namespace {

// Extra data stored for every active track
struct track_info_t
{
  // Track ID for the given track this struct extends
  track_id_t tid;

  // Location of this track in the reference frame
  vector_2d ref_loc;

  // Is the ref loc valid?
  bool ref_loc_valid;

  // Reference frame ID
  frame_id_t ref_id;

  // Does this point satisfy all required backprojection properties?
  bool is_good;

  // The number of times we haven't seen this track in the active set
  size_t missed_count;

  // On the current frame was this track updated?
  bool active;

  // Pointer to the latest instance of the track containing the above id
  track_sptr trk;

  // Constructor.
  track_info_t()
    : ref_loc( 0.0, 0.0 ),
      ref_loc_valid( false ),
      is_good( true ),
      missed_count( 0 ),
      active( false )
  {}
};

// Buffer type for storing the extra track info for all tracks
typedef std::vector< track_info_t > track_info_buffer_t;

// Pointer to a track info buffer
typedef std::shared_ptr< track_info_buffer_t > track_info_buffer_sptr;

// ----------------------------------------------------------------------------
// Helper function for sorting tis
bool
compare_ti( const track_info_t& c1, const track_info_t& c2 )
{
  return ( c1.tid < c2.tid );
}

// ----------------------------------------------------------------------------
// Find a track in a given buffer
track_info_buffer_t::iterator
find_track( const track_sptr& trk, track_info_buffer_sptr buffer )
{
  track_info_t ti;
  ti.tid = trk->id();

  auto end = buffer->end();
  auto result = std::lower_bound( buffer->begin(), end, ti, compare_ti );
  // Check that we really found the track_info_t we're looking for
  return ( result == end || result->tid == ti.tid ) ? result : end;
}

// ----------------------------------------------------------------------------
// Reset all is found flags
void
reset_active_flags( track_info_buffer_sptr buffer )
{
  for( track_info_t& ti : *buffer )
  {
    ti.active = false;
  }
}

} // end namespace anonymous

// Private implementation class
class compute_ref_homography_core::priv
{
public:
  priv( compute_ref_homography_core& parent )
    : parent( parent ),
      frames_since_reset( 0 ),
      min_ref_frame( 0 )
  {}

  compute_ref_homography_core& parent;

  // Configuration values
  double c_use_backproject_error() { return parent.c_use_backproject_error; }

  double
  c_backproject_threshold_sqr()
  {
    return parent.c_backproject_threshold_sqr;
  }

  size_t
  c_forget_track_threshold()
  {
    return parent.c_forget_track_threshold;
  }

  size_t c_min_track_length() { return parent.c_min_track_length; }
  double
  c_inlier_scale() const { return parent.c_inlier_scale; }
  size_t
  c_minimum_inliers() const { return parent.c_minimum_inliers; }

  bool
  c_allow_ref_frame_regression()
  {
    return parent.c_allow_ref_frame_regression;
  }

  vital::algo::estimate_homography_sptr
  estimator() const
  {
    return parent.c_estimator;
  }

  // Local values

  /// Buffer storing track extensions
  track_info_buffer_sptr buffer;

  /// Number of frames since last new reference frame declared
  size_t frames_since_reset;

  /// If we should allow reference frame regression or not when determining the
  /// earliest reference frame of active tracks.
  bool allow_ref_frame_regression;

  /// Minimum allowable reference frame. This is updated when homography
  /// estimation fails.
  frame_id_t min_ref_frame;

  /// Estimate the homography between two corresponding points sets
  ///
  /// Check for homography validity.
  ///
  /// Output homography describes transformation from pts_src to pts_dst.
  ///
  /// If estimate homography is deemed bad, true is returned and the
  /// homography passed to \p out_h is not modified. If false is returned, the
  /// computed homography is valid and out_h is set to the estimated homography.
  bool
  compute_homography(
    std::vector< vector_2d > const& pts_src,
    std::vector< vector_2d > const& pts_dst,
    homography_sptr& out_h ) const
  {
    bool is_bad_homog = false;
    homography_sptr tmp_h;

    // Make sure that we have at least the minimum number of points to match
    // between source and destination
    if( pts_src.size() < this->c_minimum_inliers() ||
        pts_dst.size() < this->c_minimum_inliers() )
    {
      LOG_WARN(
        parent.logger(),
        "Insufficient point pairs given to match. " <<
          "Given " << pts_src.size() << " but require at least " <<
          this->c_minimum_inliers() );
      is_bad_homog = true;
    }
    else
    {
      std::vector< bool > inliers;
      tmp_h = this->estimator()->estimate(
        pts_src, pts_dst, inliers,
        this->c_inlier_scale() );

      // Check for positive inlier count
      size_t inlier_count = 0;
      for( bool b : inliers )
      {
        if( b )
        {
          ++inlier_count;
        }
      }
      LOG_INFO(
        parent.logger(),
        "Inliers after estimation: " << inlier_count );
      if( inlier_count < this->c_minimum_inliers() )
      {
        LOG_WARN(
          parent.logger(),
          "Insufficient inliers after estimation. Require " <<
            this->c_minimum_inliers() );
        is_bad_homog = true;
      }
    }

    // Checking homography output for invertability and invalid values
    // Only need to try this if a supposed valid homog was estimated above
    if( !is_bad_homog )
    {
      try
      {
        // Invertible test
        Eigen::Matrix< double, 3, 3 > h_mat = tmp_h->matrix(),
          i_mat = tmp_h->inverse()->matrix();
        if( !( h_mat.allFinite() && i_mat.allFinite() ) )
        {
          LOG_WARN(
            parent.logger(),
            "Found non-finite values in estimated homography. Bad homography." );
          is_bad_homog = true;
        }
      }
      catch( ... )
      {
        LOG_WARN(
          parent.logger(),
          "Homography non-invertable. Bad homography." );
        is_bad_homog = true;
      }
    }

    if( !is_bad_homog )
    {
      out_h = tmp_h;
    }

    return is_bad_homog;
  }
};

// ----------------------------------------------------------------------------
void
compute_ref_homography_core
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d_ );
  attach_logger( "arrows.core.compute_ref_homography_core" );
}

compute_ref_homography_core
::~compute_ref_homography_core()
{}

// ----------------------------------------------------------------------------
bool
compute_ref_homography_core
::check_configuration( vital::config_block_sptr config ) const
{
  return
    (
    check_nested_algo_configuration< algo::estimate_homography >(
      "estimator",
      config )
    );
}

// ----------------------------------------------------------------------------
// Perform actual current to reference frame estimation
f2f_homography_sptr
compute_ref_homography_core
::estimate(
  frame_id_t frame_number,
  feature_track_set_sptr tracks ) const
{
  LOG_DEBUG(
    logger(),
    "Starting ref homography estimation for frame " << frame_number );

  // Get active tracks for the current frame
  std::vector< track_sptr > active_tracks =
    tracks->active_tracks( frame_number );

  // This is either the first frame, or a new reference frame
  if( !d_->buffer )
  {
    d_->buffer = track_info_buffer_sptr( new track_info_buffer_t() );
    d_->frames_since_reset = 0;
  }

  track_info_buffer_sptr new_buffer( new track_info_buffer_t() );
  std::vector< track_sptr > new_tracks;
  reset_active_flags( d_->buffer );

  // Flag tracks on this frame as new tracks, or "active" tracks, or tracks
  // that are not new.
  for( track_sptr trk : active_tracks )
  {
    track_info_buffer_t::iterator p = find_track( trk, d_->buffer );

    // The track was active
    if( p != d_->buffer->end() )
    {
      p->active = true;
      p->missed_count = 0;
      p->trk = trk;
    }
    else
    {
      new_tracks.push_back( trk );
    }
  }
  LOG_DEBUG(
    logger(),
    active_tracks.size() << " tracks on current frame (" <<
      ( active_tracks.size() - new_tracks.size() ) << " active, " <<
      new_tracks.size() << " new)" );

  // Add active tracks to new buffer, skipping those that we haven't seen in
  // a while.
  frame_id_t earliest_ref = std::numeric_limits< frame_id_t >::max();

  for( track_info_t& ti : *( d_->buffer ) )
  {
    if( ti.active || ++ti.missed_count < d_->c_forget_track_threshold() )
    {
      new_buffer->push_back( ti );
    }

    // Save earliest reference frame of active tracks
    // If not allowing regression, take max against min_ref_frame
    if( ti.active && ti.ref_id < earliest_ref &&
        ( d_->allow_ref_frame_regression ||
          ( ti.ref_id >= d_->min_ref_frame ) ) )
    {
      earliest_ref = ti.ref_id;
    }
  }
  LOG_DEBUG(
    logger(),
    "Earliest Ref: " << earliest_ref );

  // Add new tracks to buffer.
  for( track_sptr trk : new_tracks )
  {
    track::history_const_itr itr = trk->find( frame_number );
    if( itr == trk->end() )
    {
      continue;
    }

    auto fts = std::dynamic_pointer_cast< feature_track_state >( *itr );
    if( fts && fts->feature )
    {
      track_info_t new_entry;

      new_entry.tid = trk->id();
      new_entry.ref_loc = fts->feature->loc();
      new_entry.ref_id = frame_number;
      new_entry.active = false; // don't want to use this track on this frame
      new_entry.trk = trk;

      new_buffer->push_back( new_entry );
    }
  }

  // Ensure that the vector is still sorted. Chances are it still is and
  // this is a simple linear scan of the vector to ensure this.
  // This is needed for the find_track function's use of std::lower_bound
  // to work as expected.
  std::sort( new_buffer->begin(), new_buffer->end(), compare_ti );

  // Generate points to feed into homography regression
  std::vector< vector_2d > pts_ref, pts_cur;

  // Accept tracks that either stretch back to the reset point, or satisfy the
  // minimum track length parameter.
  size_t track_size_thresh = std::min(
    d_->c_min_track_length(),
    d_->frames_since_reset + 1 );

  // Collect cur/ref points from track infos that have earliest-frame references
  for( track_info_t& ti : *new_buffer )
  {
    // If the track is active and have a state on the earliest ref frame,
    // also include those points for homography estimation.
    if( ti.active && ti.is_good &&
        ti.ref_id == earliest_ref &&
        ti.trk->size() >= track_size_thresh )
    {
      track::history_const_itr itr = ti.trk->find( frame_number );

      auto fts = std::dynamic_pointer_cast< feature_track_state >( *itr );
      if( fts && fts->feature )
      {
        pts_ref.push_back( ti.ref_loc );
        pts_cur.push_back( fts->feature->loc() );
      }
    }
  }
  LOG_DEBUG(
    logger(),
    "Using " << pts_ref.size() << " points for estimation" );

  // Compute homography if possible
  homography_sptr h; // raw homography transform
  bool bad_homog = d_->compute_homography( pts_cur, pts_ref, h );

  // If the homography is bad, output an identity
  f2f_homography_sptr output;

  if( bad_homog )
  {
    LOG_DEBUG( logger(), "estimation FAILED" );
    // Start of new shot. Both frames the same and identity transform.
    output = f2f_homography_sptr( new f2f_homography( frame_number ) );
    d_->frames_since_reset = 0;
    d_->min_ref_frame = frame_number;
  }
  else
  {
    LOG_DEBUG( logger(), "estimation SUCCEEDED" );
    // extend current shot
    h = h->normalize();
    output = f2f_homography_sptr(
      new f2f_homography(
        h, frame_number,
        earliest_ref ) );
  }

  // Update track infos based on homography estimation result
  //  - With a valid homography, transform the reference location of active
  //    tracks with a different reference frame than the current earliest_ref
  size_t ti_reset_count = 0;
  for( track_info_t& ti : *new_buffer )
  {
    track::history_const_itr itr = ti.trk->find( frame_number );

    // skip updating track items for tracks that don't have a state on this
    // frame, or a state without a feature (location)
    if( itr == ti.trk->end() )
    {
      continue;
    }

    auto fts = std::dynamic_pointer_cast< feature_track_state >( *itr );
    if( !fts || !fts->feature )
    {
      continue;
    }

    if( !bad_homog )
    {
      // Update reference locations of active tracks that don't point to the
      // earliest_ref, and tracks that were just initialized (ref_id =
      // current_frame).
      if( ( ti.active && ti.ref_id != earliest_ref ) ||
          ti.ref_id == frame_number )
      {
        ti.ref_loc = output->homography()->map( fts->feature->loc() );
        ti.ref_id = output->to_id();
      }
      // Test back-projection on active tracks that we did not just set ref_loc
      // of.
      else if( d_->c_use_backproject_error() && ti.active )
      {
        vector_2d warped = output->homography()->map( fts->feature->loc() );
        double dist_sqr = ( warped - ti.ref_loc ).squaredNorm();

        if( dist_sqr > d_->c_backproject_threshold_sqr() )
        {
          ti.is_good = false;
        }
      }
    }
    // If not allowing ref regression, update reference loc and id of
    // active tracks to the current frame on estimation failure.
    else if( !d_->allow_ref_frame_regression && ti.active )
    {
      ++ti_reset_count;
      ti.ref_loc = fts->feature->loc();
      ti.ref_id = frame_number;
    }
  }

  if( IS_DEBUG_ENABLED( logger() ) &&  ti_reset_count )
  {
    LOG_DEBUG(
      logger(),
      "Resetting " << ti_reset_count <<
        " tracks to reference frame: " << frame_number );
  }

  // Increment counter, update buffers
  d_->frames_since_reset++;
  d_->buffer = new_buffer;

  return output;
}

} // end namespace core

} // end namespace arrows

} // end namespace kwiver
