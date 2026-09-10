// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of initialize_object_tracks_threshold

#include "initialize_object_tracks_threshold.h"

#include <viame/algorithm_framework/algo/detected_object_filter.h>
#include <viame/algorithm_framework/exceptions/algorithm.h>
#include <viame/core_types/object_track_set.h>

#include <algorithm>
#include <atomic>
#include <string>
#include <vector>

namespace kwiver {

namespace arrows {

namespace core {

using namespace kwiver::vital;

/// Private implementation class
class initialize_object_tracks_threshold::priv
{
public:
  /// Constructor
  priv( initialize_object_tracks_threshold& parent )
    : parent( parent )
  {}

  initialize_object_tracks_threshold& parent;

  /// Maximum number of tracks to initialize
  size_t c_max_new_tracks() { return parent.c_max_new_tracks; }

  /// The feature matching algorithm to use
  vital::algo::detected_object_filter_sptr
  c_filter()
  {
    return parent.c_filter;
  }

  /// Next track ID to assign - make unique across all processes
  static std::atomic< size_t > next_track_id;
};

// Initialize statics
std::atomic< size_t >
initialize_object_tracks_threshold::priv::next_track_id( 1 );

void
initialize_object_tracks_threshold
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d_ );
  attach_logger( "arrows.core.initialize_object_tracks_threshold" );
}

/// Destructor
initialize_object_tracks_threshold
::~initialize_object_tracks_threshold() noexcept
{}

bool
initialize_object_tracks_threshold
::check_configuration( vital::config_block_sptr config ) const
{
  return (
    check_nested_algo_configuration< algo::detected_object_filter >(
      "filter",
      config )
  );
}

/// Initialize object tracks
kwiver::vital::object_track_set_sptr
initialize_object_tracks_threshold
::initialize(
  kwiver::vital::timestamp ts,
  kwiver::vital::image_container_sptr /*image*/,
  kwiver::vital::detected_object_set_sptr detections ) const
{
  auto filtered = d_->c_filter()->filter( detections );
  std::vector< vital::track_sptr > output;

  size_t max_tracks = std::min(
    static_cast< size_t >( filtered->size() ),
    d_->c_max_new_tracks() );

  for( size_t i = 0; i < max_tracks; i++ )
  {
    size_t new_id = initialize_object_tracks_threshold::priv::next_track_id++;

    vital::track_sptr new_track( vital::track::create() );
    new_track->set_id( new_id );

    vital::track_state_sptr first_track_state(
      new vital::object_track_state( ts, filtered->at( i ) ) );

    new_track->append( first_track_state );

    output.push_back( new_track );
  }

  return vital::object_track_set_sptr( new object_track_set( output ) );
}

} // end namespace core

} // end namespace arrows

} // end namespace kwiver
