/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Consolidate the output of multiple object trackers
 */

#include "full_frame_tracker_process.h"

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

create_config_trait( fixed_frame_count, unsigned, "0",
  "If set, generate a full frame track for this many frames" );

// =============================================================================
// Private implementation class
class full_frame_tracker_process::priv
{
public:
  explicit priv( full_frame_tracker_process* parent );
  ~priv();

  // Configuration settings
  unsigned m_fixed_frame_count;

  // Internal variables
  unsigned m_frame_counter;
  unsigned m_track_counter;
  std::vector< kv::track_state_sptr > m_states;

  // Other variables
  full_frame_tracker_process* parent;
};


// -----------------------------------------------------------------------------
full_frame_tracker_process::priv
::priv( full_frame_tracker_process* ptr )
  : m_fixed_frame_count( 0 )
  , m_frame_counter( 0 )
  , m_track_counter( 1 )
  , parent( ptr )
{
}


full_frame_tracker_process::priv
::~priv()
{
}


// =============================================================================
full_frame_tracker_process
::full_frame_tracker_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new full_frame_tracker_process::priv( this ) )
{
  make_ports();
  make_config();
}


full_frame_tracker_process
::~full_frame_tracker_process()
{
}


// -----------------------------------------------------------------------------
void
full_frame_tracker_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( image, optional );
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( detected_object_set, required );

  // -- outputs --
  declare_output_port_using_trait( timestamp, optional );
  declare_output_port_using_trait( object_track_set, optional );
}

// -----------------------------------------------------------------------------
void
full_frame_tracker_process
::make_config()
{
  declare_config_using_trait( fixed_frame_count );
}

// -----------------------------------------------------------------------------
void
full_frame_tracker_process
::_configure()
{
  d->m_fixed_frame_count = config_value_using_trait( fixed_frame_count );
}

// -----------------------------------------------------------------------------
void
full_frame_tracker_process
::_step()
{
  kv::image_container_sptr image;
  kv::timestamp timestamp;
  kv::detected_object_set_sptr detections;

  if( has_input_port_edge_using_trait( timestamp ) )
  {
    timestamp = grab_from_port_using_trait( timestamp );
  }
  if( has_input_port_edge_using_trait( image ) )
  {
    image = grab_from_port_using_trait( image );
  }
  if( has_input_port_edge_using_trait( detected_object_set ) )
  {
    detections = grab_from_port_using_trait( detected_object_set );
  }

  if( d->m_states.size() == d->m_fixed_frame_count )
  {
    d->m_track_counter++;
    d->m_states.clear();
  }

  if( detections->size() == 1 )
  {
    d->m_states.push_back(
      std::make_shared< kwiver::vital::object_track_state >(
        timestamp, detections->at( 0 ) ) );
  }

  kv::track_sptr ot = kv::track::create();
  ot->set_id( d->m_track_counter );

  for( auto state : d->m_states )
  {
    ot->append( state );
  }

  kv::object_track_set_sptr output(
    new kv::object_track_set(
      std::vector< kv::track_sptr >( 1, ot ) ) );

  push_to_port_using_trait( timestamp, timestamp );
  push_to_port_using_trait( object_track_set, output );
}

} // end namespace core

} // end namespace viame
