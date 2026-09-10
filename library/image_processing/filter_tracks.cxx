// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of core filter_tracks algorithm
#include <viame/image_processing/filter_tracks.h>
#include <viame/image_processing/match_matrix.h>

#include <algorithm>

using namespace kwiver::vital;

namespace kwiver {

namespace arrows {

namespace core {

/// Private implementation class
class filter_tracks::priv
{
public:
  priv( filter_tracks& parent )
    : parent( parent )
  {}

  filter_tracks& parent;

  // Configuration values
  size_t c_min_track_length() { return parent.c_min_track_length; }
  double c_min_mm_importance() { return parent.c_min_mm_importance; }
};

// ----------------------------------------------------------------------------
void
filter_tracks
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d_ );
  attach_logger( "arrows.core.filter_tracks" );
}

// Destructor
filter_tracks
::~filter_tracks()
{}

// ----------------------------------------------------------------------------
// Check that the algorithm's configuration vital::config_block is valid
bool
filter_tracks
::check_configuration( vital::config_block_sptr config ) const
{
  double min_mm_importance = config->get_value< double >(
    "min_mm_importance",
    d_->c_min_mm_importance() );
  if( min_mm_importance < 0.0 )
  {
    LOG_ERROR(
      logger(),
      "min_mm_importance parameter is " << min_mm_importance
                                        << ", must be non-negative." );
    return false;
  }
  return true;
}

// ----------------------------------------------------------------------------
// Filter feature set
vital::track_set_sptr
filter_tracks
::filter( vital::track_set_sptr tracks ) const
{
  if( d_->c_min_track_length() > 1 )
  {
    std::vector< kwiver::vital::track_sptr > trks = tracks->tracks();
    std::vector< kwiver::vital::track_sptr > good_trks;
    for( kwiver::vital::track_sptr t : trks )
    {
      if( t->size() >= d_->c_min_track_length() )
      {
        good_trks.push_back( t );
      }
    }
    tracks = std::make_shared< kwiver::vital::track_set >(
      good_trks,
      tracks->all_frame_data() );
  }

  if( d_->c_min_mm_importance() > 0 )
  {
    // compute the match matrix
    std::vector< vital::frame_id_t > frames;
    Eigen::SparseMatrix< size_t > mm =
      kwiver::arrows::match_matrix( tracks, frames );

    // compute the importance scores on the tracks
    std::map< vital::track_id_t, double > importance =
      kwiver::arrows::match_matrix_track_importance( tracks, frames, mm );

    std::vector< kwiver::vital::track_sptr > trks = tracks->tracks();
    std::vector< vital::track_sptr > good_trks;
    for( kwiver::vital::track_sptr t : trks )
    {
      std::map< vital::track_id_t, double >::const_iterator itr;
      if( ( itr = importance.find( t->id() ) ) != importance.end() &&
          itr->second >= d_->c_min_mm_importance() )
      {
        good_trks.push_back( t );
      }
    }

    tracks = std::make_shared< kwiver::vital::track_set >(
      good_trks,
      tracks->all_frame_data() );
  }
  return tracks;
}

} // end namespace core

} // end namespace arrows

} // end namespace kwiver
