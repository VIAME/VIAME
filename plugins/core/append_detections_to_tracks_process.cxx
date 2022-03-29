/**
 * \file
 * \brief Append a detected object set to an object track set
 */

#include "append_detections_to_tracks_process.h"

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
  "If set, generate an appended object track set for this many detected object set" );

// =============================================================================
// Private implementation class
class append_detections_to_tracks_process::priv
{
public:
  explicit priv( append_detections_to_tracks_process* parent );
  ~priv();

  // Configuration settings
  unsigned m_fixed_frame_count;
  
  // Internal variables
  unsigned m_frame_counter;
  unsigned m_track_counter;
  std::vector<std::vector< kv::track_state_sptr >> m_states;

  // Other variables
  append_detections_to_tracks_process* parent;
};


// -----------------------------------------------------------------------------
append_detections_to_tracks_process::priv
::priv( append_detections_to_tracks_process* ptr )
  : m_fixed_frame_count( 0 )
  , m_frame_counter( 0 )
  , m_track_counter( 1 )
  , parent( ptr )
{
}


append_detections_to_tracks_process::priv
::~priv()
{
}


// =============================================================================
append_detections_to_tracks_process
::append_detections_to_tracks_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new append_detections_to_tracks_process::priv( this ) )
{
  make_ports();
  make_config();
}


append_detections_to_tracks_process
::~append_detections_to_tracks_process()
{
}


// -----------------------------------------------------------------------------
void
append_detections_to_tracks_process
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
append_detections_to_tracks_process
::make_config()
{
  declare_config_using_trait( fixed_frame_count );
}

// -----------------------------------------------------------------------------
void
append_detections_to_tracks_process
::_configure()
{
  d->m_fixed_frame_count = config_value_using_trait( fixed_frame_count );
}

// -----------------------------------------------------------------------------
void
append_detections_to_tracks_process
::_step()
{
  kv::image_container_sptr image;
  kv::timestamp timestamp;
  kv::detected_object_set_sptr detections;
  kv::object_track_set_sptr output;
  
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
  
  if(!d->m_frame_counter && detections->size()) 
  {
    d->m_states.resize(detections->size());
  }

  if( !d->m_fixed_frame_count || d->m_frame_counter <= d->m_fixed_frame_count)
  {
    if( d->m_states.size() &&  d->m_states.size() == detections->size())
    {
      std::vector< kv::track_sptr > all_tracks;
      for( unsigned detectId = 0; detectId < detections->size(); ++detectId )
      {
        d->m_states[detectId].push_back(
              std::make_shared< kwiver::vital::object_track_state >(
                timestamp, detections->at( detectId ) ) );
        
        kv::track_sptr ot = kv::track::create();
        ot->set_id( detectId );
        
        for( auto state : d->m_states[detectId] )
        {
          ot->append( state );
        }
        
        all_tracks.push_back(ot);
      }
      
      output = kv::object_track_set_sptr(new kv::object_track_set(all_tracks));
      
      d->m_frame_counter++;
    }
  }
  
  push_to_port_using_trait( timestamp, timestamp );
  push_to_port_using_trait( object_track_set, output );
}

} // end namespace core

} // end namespace viame
