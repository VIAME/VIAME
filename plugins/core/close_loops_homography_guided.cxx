/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Loop closure guided by ground plane homographies

#include "close_loops_homography_guided.h"

#include <algorithm>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>

#include <vital/algo/algorithm.txx>
#include <vital/algo/compute_ref_homography.h>
#include <vital/algo/match_features.h>
#include <vital/vital_config.h>

#include <image_ops/homography_overlap.h>

using namespace kwiver::vital;

namespace viame {

namespace {

// Data stored for every detected checkpoint
class checkpoint_entry_t
{
public:
  // Constructor
  checkpoint_entry_t()
    : fid( 0 ),
      src_to_ref( new f2f_homography( fid ) )
  {}

  // Constructor
  checkpoint_entry_t( frame_id_t id, f2f_homography_sptr h )
    : fid( id ),
      src_to_ref( h )
  {}

  // Frame ID of checkpoint
  frame_id_t fid;

  // Source to ref homography
  f2f_homography_sptr src_to_ref;
};

// Buffer type for detected checkpoints
typedef std::deque< checkpoint_entry_t > checkpoint_buffert;

// Buffer reverse iterator
typedef checkpoint_buffert::reverse_iterator bufferritr;

// If possible convert a src1 to ref and src2 to ref homography to a src2 to
// src1 homography
bool
convert(
  const f2f_homography_sptr& src1_to_ref,
  const f2f_homography_sptr& src2_to_ref,
  Eigen::Matrix< double, 3, 3 >& src2_to_src1 )
{
  try
  {
    src2_to_src1 = ( src1_to_ref->inverse() *
                     ( *src2_to_ref ) ).homography()->matrix();
    return true;
  }
  catch( ... )
  {
    kwiver::vital::logger_handle_t logger( kwiver::vital::get_logger(
      "viame.core.close_loops_homography_guided" ) );
    LOG_ERROR(logger, "Warn: Invalid homography received" );
  }

  src2_to_src1 = src2_to_ref->homography()->matrix();
  return false;
}

} // end namespace anonymous

/// Private implementation class
class close_loops_homography_guided::priv
{
public:
  priv( close_loops_homography_guided& parent ) : parent{ parent } {}

  close_loops_homography_guided& parent;

  /// Is long term loop closure enabled?
  bool
  c_enabled() const { return parent.c_enabled; }

  /// Maximum past search distance in terms of number of checkpoints.
  unsigned
  c_max_checkpoint_frames() const { return parent.c_max_checkpoint_frames; }

  /// Term which controls when we make new loop closure checkpoints.
  double
  c_checkpoint_percent_overlap() const
  {
    return parent.c_checkpoint_percent_overlap;
  }

  /// Output filename for homographies
  std::string
  c_homography_filename() const { return parent.c_homography_filename; }

  /// Buffer storing past homographies for checkpoint frames
  checkpoint_buffert buffer;

  /// Reference frame homography computer
  kwiver::vital::algo::compute_ref_homography_sptr ref_computer;

  /// The feature matching algorithm to use
  kwiver::vital::algo::match_features_sptr matcher;
};

// ----------------------------------------------------------------------------
void
close_loops_homography_guided
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d );
  attach_logger( "viame.core.close_loops_homography_guided" );
}

// ----------------------------------------------------------------------------
bool
close_loops_homography_guided
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return
    (
    kwiver::vital::check_nested_algo_configuration< kwiver::vital::algo::compute_ref_homography >(
      "ref_computer", config )
    &&
    kwiver::vital::check_nested_algo_configuration< kwiver::vital::algo::match_features >(
      "feature_matcher", config )
    );
}

// ----------------------------------------------------------------------------
// Perform stitch operation
feature_track_set_sptr
close_loops_homography_guided
::stitch(
  frame_id_t frame_number,
  feature_track_set_sptr input,
  image_container_sptr image,
  [[maybe_unused]] image_container_sptr mask ) const
{
  if( !d->c_enabled() || !image )
  {
    return input;
  }

  auto const width = image->width();
  auto const height = image->height();

  // Compute new homographies for this frame (current_to_ref)
  f2f_homography_sptr homog = d->ref_computer->estimate(
    frame_number,
    input );

  // Write out homographies if enabled
  if( !d->c_homography_filename().empty() )
  {
    std::ofstream fout( d->c_homography_filename().c_str(), std::ios::app );
    fout << *homog << std::endl;
    fout.close();
  }

  // Determine if this is a new checkpoint frame. Either the buffer is empty
  // and this is a new frame, this is a homography for a new reference frame,
  // or the overlap with the last checkpoint was less than a specified amount.
  Eigen::Matrix< double, 3, 3 > tmp;

  if( d->buffer.empty() ||
      !convert( d->buffer.back().src_to_ref, homog, tmp ) ||
      viame::image_ops::homography_overlap(
        tmp.data(), width,
        height ) < d->c_checkpoint_percent_overlap() )
  {
    d->buffer.push_back( checkpoint_entry_t( frame_number, homog ) );
    if( d->buffer.size() > d->c_max_checkpoint_frames() )
    {
      d->buffer.pop_front();
    }
  }

  // Perform matching to any past checkpoints we want to test
  enum { initial, non_intersection, reintersection, } scan_state;

  bufferritr best_frame_to_test = d->buffer.rend();
  double best_frame_intersection = 0.0;
  scan_state = initial;

  for( bufferritr itr = d->buffer.rbegin(); itr != d->buffer.rend();
       itr++ )
  {
    bool transform_valid = convert( itr->src_to_ref, homog, tmp );
    double po = 0.0;

    if( transform_valid )
    {
      po = viame::image_ops::homography_overlap(
        tmp.data(), width, height );
    }

    if( scan_state == reintersection )
    {
      if( !transform_valid || !po )
      {
        break;
      }

      if( po > best_frame_intersection )
      {
        best_frame_to_test = itr;
        best_frame_intersection = po;
      }
    }
    else if( scan_state == initial )
    {
      if( !transform_valid || !po )
      {
        scan_state = non_intersection;
      }
    }
    else if( transform_valid && po > 0 )
    {
      best_frame_to_test = itr;
      best_frame_intersection = po;
      scan_state = reintersection;
    }
  }

// Perform matching operation to target frame if possible
  if( best_frame_to_test != d->buffer.rend() )
  {
    const frame_id_t prior_frame = best_frame_to_test->fid;

    // Perform matching operation
    match_set_sptr mset = d->matcher->match(
      input->frame_features( frame_number ),
      input->frame_descriptors( frame_number ),
      input->frame_features( prior_frame ),
      input->frame_descriptors( prior_frame ) );

    // Test matcher results
    if( mset->size() > 0 ) // If matches are good
    {
      // Logging
      LOG_INFO(
        logger(),
        "Stitching frames " << prior_frame << " and " << frame_number);

      // Get all tracks on the past frame
      std::vector< track_sptr > prior_trks =
        input->active_tracks( prior_frame );

      // Get all tracks on the current frame
      std::vector< track_sptr > current_trks =
        input->active_tracks( frame_number );

      // Get all matches
      std::vector< match > matches = mset->matches();

      for( unsigned i = 0; i < matches.size(); i++ )
      {
        input->merge_tracks(
          current_trks[ matches[ i ].first ],
          prior_trks[ matches[ i ].second ] );
      }

      // Return updated set
      return input;
    }
  }

// Return input set
  return input;
}

} // end namespace viame
