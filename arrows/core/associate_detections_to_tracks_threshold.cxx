/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

/**
 * \file
 * \brief Implementation of associate_detections_to_tracks_threshold
 */

#include "associate_detections_to_tracks_threshold.h"

#include <vital/algo/detected_object_filter.h>
#include <vital/types/object_track_set.h>
#include <vital/exceptions/algorithm.h>

#include <string>
#include <vector>
#include <atomic>
#include <algorithm>
#include <limits>


namespace kwiver {
namespace arrows {
namespace core {

using namespace kwiver::vital;


/// Private implementation class
class associate_detections_to_tracks_threshold::priv
{
public:
  /// Constructor
  priv()
    : threshold( 0.50 )
    , higher_is_better( true )
    , m_logger( vital::get_logger(
        "arrows.core.associate_detections_to_tracks_threshold" ) )
  {
  }

  /// Threshold to apply on the matrix
  double threshold;

  /// Whether to take values above or below the threshold
  bool higher_is_better;

  /// Logger handle
  vital::logger_handle_t m_logger;
};


/// Constructor
associate_detections_to_tracks_threshold
::associate_detections_to_tracks_threshold()
  : d_( new priv )
{
}


/// Destructor
associate_detections_to_tracks_threshold
::~associate_detections_to_tracks_threshold() VITAL_NOTHROW
{
}


/// Get this alg's \link vital::config_block configuration block \endlink
vital::config_block_sptr
associate_detections_to_tracks_threshold
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  config->set_value( "threshold", d_->threshold,
    "Threshold to apply on the matrix." );

  config->set_value( "higher_is_better", d_->higher_is_better,
    "Whether values above or below the threshold indicate a better fit." );

  return config;
}


/// Set this algo's properties via a config block
void
associate_detections_to_tracks_threshold
::set_configuration( vital::config_block_sptr in_config )
{
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  d_->threshold = config->get_value<double>( "threshold" );
  d_->higher_is_better = config->get_value<bool>( "higher_is_better" );
}


bool
associate_detections_to_tracks_threshold
::check_configuration(vital::config_block_sptr config) const
{
  return true;
}


/// Associate object detections to object tracks
bool
associate_detections_to_tracks_threshold
::associate( kwiver::vital::timestamp ts,
             kwiver::vital::image_container_sptr /*image*/,
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

  for( unsigned t = 0; t < all_tracks.size(); ++t )
  {
    double best_score = ( d_->higher_is_better ? -1 : 1 ) *
      std::numeric_limits<double>::max();

    unsigned best_index = std::numeric_limits< unsigned >::max();

    for( unsigned d = 0; d < all_detections->size(); ++d )
    {
      double value = matrix( t, d );

      if( d_->higher_is_better )
      {
        if( value >= d_->threshold && value > best_score )
        {
          best_score = value;
          best_index = d;
        }
      }
      else
      {
        if( value <= d_->threshold && value < best_score )
        {
          best_score = value;
          best_index = d;
        }
      }
    }

    if( best_index < all_detections->size() )
    {
      vital::track_state_sptr new_track_state(
        new vital::object_track_state( ts.get_frame(),
          all_detections->begin()[best_index] ) );

      vital::track_sptr adj_track( all_tracks[t]->clone() );
      adj_track->append( new_track_state );
      tracks_to_output.push_back( adj_track );

      detections_used[best_index] = true;
    }
    else
    {
      tracks_to_output.push_back( all_tracks[t] );
    }
  }

  std::vector< vital::detected_object_sptr > unused_dets;

  for( unsigned i = 0; i < all_detections->size(); ++i )
  {
    if( !detections_used[i] )
    {
      unused_dets.push_back( all_detections->begin()[i] );
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
