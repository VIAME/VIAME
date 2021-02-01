// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
::~initialize_object_tracks_threshold() noexcept
{
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
  auto filtered = d_->filter->filter( detections );
  std::vector< vital::track_sptr > output;

  unsigned max_tracks = std::min( static_cast<unsigned>( filtered->size() ), d_->max_new_tracks );

  for( unsigned i = 0; i < max_tracks; i++ )
  {
    unsigned new_id = initialize_object_tracks_threshold::priv::next_track_id++;

    vital::track_sptr new_track( vital::track::create() );
    new_track->set_id( new_id );

    vital::track_state_sptr first_track_state(
      new vital::object_track_state( ts, filtered->at(i) ) );

    new_track->append( first_track_state );

    output.push_back( new_track );
  }

  return vital::object_track_set_sptr( new object_track_set( output ) );
}

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
