/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Consolidate the output of multiple object trackers
 */

#include "filter_object_tracks_process.h"

#include <vital/vital_types.h>
#include <vital/types/image_container.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/object_track_set.h>

#include <sprokit/processes/kwiver_type_traits.h>


namespace kv = kwiver::vital;

namespace viame
{

namespace core
{

create_config_trait( required_states, unsigned, "0",
  "If set, number of track states required for a track"  );
create_config_trait( buffer_frames, unsigned, "0",
  "Number of frames to buffer offer"  );

// =============================================================================
// Private implementation class
class filter_object_tracks_process::priv
{
public:
  explicit priv( filter_object_tracks_process* parent );
  ~priv();

  // Configuration settings
  unsigned m_required_states;
  unsigned m_buffer_frames;

  // Other variables
  filter_object_tracks_process* parent;
};


// -----------------------------------------------------------------------------
filter_object_tracks_process::priv
::priv( filter_object_tracks_process* ptr )
  : m_required_states( 0 )
  , m_buffer_frames( 0 )
  , parent( ptr )
{
}


filter_object_tracks_process::priv
::~priv()
{
}


// =============================================================================
filter_object_tracks_process
::filter_object_tracks_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new filter_object_tracks_process::priv( this ) )
{
  make_ports();
  make_config();
}


filter_object_tracks_process
::~filter_object_tracks_process()
{
}


// -----------------------------------------------------------------------------
void
filter_object_tracks_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( image, optional );
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( object_track_set, required );

  // -- outputs --
  declare_output_port_using_trait( timestamp, optional );
  declare_output_port_using_trait( object_track_set, optional );
}

// -----------------------------------------------------------------------------
void
filter_object_tracks_process
::make_config()
{
  declare_config_using_trait( required_states );
  declare_config_using_trait( buffer_frames );
}

// -----------------------------------------------------------------------------
void
filter_object_tracks_process
::_configure()
{
  d->m_required_states = config_value_using_trait( required_states );
  d->m_buffer_frames = config_value_using_trait( buffer_frames );
}

// -----------------------------------------------------------------------------
void
filter_object_tracks_process
::_step()
{
  kv::object_track_set_sptr input_tracks;

  kv::image_container_sptr image;
  kv::timestamp timestamp;

  input_tracks = grab_from_port_using_trait( object_track_set );

  if( has_input_port_edge_using_trait( timestamp ) )
  {
    timestamp = grab_from_port_using_trait( timestamp );
  }
  if( has_input_port_edge_using_trait( image ) )
  {
    image = grab_from_port_using_trait( image );
  }

  std::vector< kv::track_sptr > filtered_tracks;

  for( auto trk : input_tracks->tracks() )
  {
    if( trk && trk->size() >= d->m_required_states )
    {
      filtered_tracks.push_back( trk );
    }
  }

  kv::object_track_set_sptr output(
    new kv::object_track_set(
      filtered_tracks ) );

  push_to_port_using_trait( timestamp, timestamp );
  push_to_port_using_trait( object_track_set, output );
}

} // end namespace core

} // end namespace viame
