// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "initialize_object_tracks_process.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>
#include <vital/types/object_track_set.h>

#include <vital/algo/initialize_object_tracks.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace kwiver {

namespace algo = vital::algo;

create_algorithm_name_config_trait( track_initializer );

//------------------------------------------------------------------------------
// Private implementation class
class initialize_object_tracks_process::priv
{
public:
  priv();
  ~priv();

  algo::initialize_object_tracks_sptr m_track_initializer;
}; // end priv class

// =============================================================================

initialize_object_tracks_process
::initialize_object_tracks_process( vital::config_block_sptr const& config )
  : process( config ),
    d( new initialize_object_tracks_process::priv )
{
  make_ports();
  make_config();
}

initialize_object_tracks_process
::~initialize_object_tracks_process()
{
}

// -----------------------------------------------------------------------------
void initialize_object_tracks_process
::_configure()
{
  scoped_configure_instrumentation();

  vital::config_block_sptr algo_config = get_config();

  algo::initialize_object_tracks::set_nested_algo_configuration_using_trait(
    track_initializer,
    algo_config,
    d->m_track_initializer );

  if( !d->m_track_initializer )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
                 "Unable to create initialize_object_tracks" );
  }

  algo::initialize_object_tracks::get_nested_algo_configuration_using_trait(
    track_initializer,
    algo_config,
    d->m_track_initializer );

  // Check config so it will give run-time diagnostic of config problems
  if( !algo::initialize_object_tracks::check_nested_algo_configuration_using_trait(
        track_initializer, algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
                 "Configuration check failed." );
  }
}

// -----------------------------------------------------------------------------
void
initialize_object_tracks_process
::_step()
{
  vital::timestamp frame_id;
  vital::image_container_sptr image;
  vital::detected_object_set_sptr detections;
  vital::object_track_set_sptr old_tracks;

  vital::object_track_set_sptr new_tracks;

  if( process::has_input_port_edge( "timestamp" ) )
  {
    frame_id = grab_from_port_using_trait( timestamp );

    // Output frame ID
    LOG_DEBUG( logger(), "Processing frame " << frame_id );
  }

  if( process::has_input_port_edge( "image" ) )
  {
    image = grab_from_port_using_trait( image );
  }

  detections = grab_from_port_using_trait( detected_object_set );

  if( process::has_input_port_edge( "object_track_set" ) )
  {
    old_tracks = grab_from_port_using_trait( object_track_set );
  }

  {
    scoped_step_instrumentation();

    // Compute new tracks
    new_tracks = d->m_track_initializer->initialize( frame_id, image, detections );
  }

  // Union optional input tracks if available
  if( old_tracks )
  {
    std::vector< vital::track_sptr > net_tracks = old_tracks->tracks();
    std::vector< vital::track_sptr > to_add = new_tracks->tracks();

    net_tracks.insert( net_tracks.end(), to_add.begin(), to_add.end() );

    vital::object_track_set_sptr joined_tracks(
      new vital::object_track_set( net_tracks ) );
    push_to_port_using_trait( object_track_set, joined_tracks );
  }
  else
  {
    push_to_port_using_trait( object_track_set, new_tracks );
  }
}

// -----------------------------------------------------------------------------
void initialize_object_tracks_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( image, optional );
  declare_input_port_using_trait( object_track_set, optional );
  declare_input_port_using_trait( detected_object_set, required );

  // -- output --
  declare_output_port_using_trait( object_track_set, optional );
}

// -----------------------------------------------------------------------------
void initialize_object_tracks_process
::make_config()
{
  declare_config_using_trait( track_initializer );
}

// -----------------------------------------------------------------------------
void initialize_object_tracks_process
::_init()
{
}

// =============================================================================
initialize_object_tracks_process::priv
::priv()
{
}

initialize_object_tracks_process::priv
::~priv()
{
}

} // end namespace
