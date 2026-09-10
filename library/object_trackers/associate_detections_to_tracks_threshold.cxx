// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of associate_detections_to_tracks_threshold

#include "associate_detections_to_tracks_threshold.h"

#include <viame/algorithm_framework/algo/detected_object_filter.h>
#include <viame/algorithm_framework/exceptions/algorithm.h>

#include <viame/core_types/object_track_set.h>
#include <viame/algorithm_framework/vital_config.h>

#include <algorithm>
#include <atomic>
#include <limits>
#include <string>
#include <vector>

namespace kwiver {

namespace arrows {

namespace core {

using namespace kwiver::vital;

/// Private implementation class
class associate_detections_to_tracks_threshold::priv
{
public:
  priv( associate_detections_to_tracks_threshold& parent )
    : parent( parent )
  {}

  associate_detections_to_tracks_threshold& parent;

  // Configuration values
  double c_threshold() { return parent.c_threshold; }
  bool c_higher_is_better() { return parent.c_higher_is_better; }
};

// ----------------------------------------------
void
associate_detections_to_tracks_threshold
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d_ );
  attach_logger( "arrows.core.associate_detections_to_tracks_threshold" );
}

/// Destructor
associate_detections_to_tracks_threshold
::~associate_detections_to_tracks_threshold() noexcept
{}

bool
associate_detections_to_tracks_threshold
::check_configuration( [[maybe_unused]] vital::config_block_sptr config ) const
{
  return true;
}

/// Associate object detections to object tracks
bool
associate_detections_to_tracks_threshold
::associate(
  kwiver::vital::timestamp ts,
  kwiver::vital::image_container_sptr /* image */,
  kwiver::vital::object_track_set_sptr tracks,
  kwiver::vital::detected_object_set_sptr detections,
  kwiver::vital::matrix_d matrix,
  kwiver::vital::object_track_set_sptr& output,
  kwiver::vital::detected_object_set_sptr& unused ) const
{
  auto all_detections = detections;
  auto all_tracks = tracks->tracks();

  std::vector< vital::track_sptr > tracks_to_output;
  std::vector< bool > detections_used( all_detections->size(), false );

  for( size_t t = 0; t < all_tracks.size(); ++t )
  {
    double best_score = ( d_->c_higher_is_better() ? -1 : 1 ) *
                        std::numeric_limits< double >::max();

    size_t best_index = std::numeric_limits< size_t >::max();

    for( size_t d = 0; d < all_detections->size(); ++d )
    {
      double value = matrix( t, d );

      if( d_->c_higher_is_better() )
      {
        if( value >= d_->c_threshold() && value > best_score )
        {
          best_score = value;
          best_index = d;
        }
      }
      else
      {
        if( value <= d_->c_threshold() && value < best_score )
        {
          best_score = value;
          best_index = d;
        }
      }
    }

    if( best_index < all_detections->size() )
    {
      vital::track_state_sptr new_track_state(
        new vital::object_track_state(
          ts,
          all_detections->at( best_index ) ) );

      vital::track_sptr adj_track( all_tracks[ t ]->clone() );
      adj_track->append( new_track_state );
      tracks_to_output.push_back( adj_track );

      detections_used[ best_index ] = true;
    }
    else
    {
      tracks_to_output.push_back( all_tracks[ t ] );
    }
  }

  std::vector< vital::detected_object_sptr > unused_dets;

  for( size_t i = 0; i < all_detections->size(); ++i )
  {
    if( !detections_used[ i ] )
    {
      unused_dets.push_back( all_detections->at( i ) );
    }
  }

  output = vital::object_track_set_sptr(
    new object_track_set( tracks_to_output ) );
  unused = vital::detected_object_set_sptr(
    new vital::detected_object_set( unused_dets ) );

  return ( unused->size() != all_detections->size() );
}

} // end namespace core

} // end namespace arrows

} // end namespace kwiver
