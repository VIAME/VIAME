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
 * \brief Implementation of initialize_object_tracks_threshold
 */

#include "initialize_object_tracks_threshold.h"

#include <vital/algo/detected_object_filter.h>
#include <vital/types/object_track_set.h>
#include <vital/exceptions/algorithm.h>

#include <string>
#include <vector>
#include <atomic>
#include <algorithm>


namespace kwiver {
namespace arrows {
namespace core {

using namespace kwiver::vital;


/// Private implementation class
class initialize_object_tracks_threshold::priv
{
public:
  /// Constructor
  priv()
    : max_new_tracks( 10000 )
    , m_logger( vital::get_logger( "arrows.core.initialize_object_tracks_threshold" ))
  {
  }

  /// Maximum number of tracks to initialize
  unsigned max_new_tracks;

  /// Next track ID to assign - make unique across all processes
  static std::atomic< unsigned > next_track_id;

  /// The feature matching algorithm to use
  vital::algo::detected_object_filter_sptr filter;

  /// Logger handle
  vital::logger_handle_t m_logger;
};


// Initialize statics
std::atomic< unsigned >
initialize_object_tracks_threshold::priv::next_track_id( 1 );


/// Constructor
initialize_object_tracks_threshold
::initialize_object_tracks_threshold()
  : d_( new priv )
{
}


/// Destructor
initialize_object_tracks_threshold
::~initialize_object_tracks_threshold() VITAL_NOTHROW
{
}


std::string
initialize_object_tracks_threshold
::description() const
{
  return "Initializes new object tracks via simple thresholding";
}


/// Get this alg's \link vital::config_block configuration block \endlink
vital::config_block_sptr
initialize_object_tracks_threshold
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  // Sub-algorithm implementation name + sub_config block
  // - Feature filter algorithm
  algo::detected_object_filter::get_nested_algo_configuration(
    "filter", config, d_->filter);

  config->set_value( "max_new_tracks", d_->max_new_tracks,
    "Maximum number of new tracks to initialize on a single frame." );

  return config;
}


/// Set this algo's properties via a config block
void
initialize_object_tracks_threshold
::set_configuration( vital::config_block_sptr in_config )
{
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  algo::detected_object_filter::set_nested_algo_configuration( "filter",
    config, d_->filter );

  d_->max_new_tracks = config->get_value<unsigned>( "max_new_tracks" );
}


bool
initialize_object_tracks_threshold
::check_configuration(vital::config_block_sptr config) const
{
  return (
    algo::detected_object_filter::check_nested_algo_configuration( "filter", config )
  );
}


/// Initialize object tracks
kwiver::vital::object_track_set_sptr
initialize_object_tracks_threshold
::initialize( kwiver::vital::timestamp ts,
              kwiver::vital::image_container_sptr /*image*/,
              kwiver::vital::detected_object_set_sptr detections ) const
{
  auto filtered = d_->filter->filter( detections )->select();
  std::vector< vital::track_sptr > output;

  unsigned max_tracks = std::min( static_cast<unsigned>( filtered.size() ), d_->max_new_tracks );

  for( unsigned i = 0; i < max_tracks; i++ )
  {
    unsigned new_id = initialize_object_tracks_threshold::priv::next_track_id++;

    vital::track_sptr new_track( vital::track::make() );
    new_track->set_id( new_id );

    vital::track_state_sptr first_track_state(
      new vital::object_track_state( ts.get_frame(), filtered[i] ) );

    new_track->append( first_track_state );

    output.push_back( new_track );
  }

  return vital::object_track_set_sptr( new object_track_set( output ) );
}


} // end namespace core
} // end namespace arrows
} // end namespace kwiver
