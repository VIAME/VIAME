/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Consolidate the output of multiple object trackers
 */

#include "track_conductor_process.h"

#include <list>
#include <limits>
#include <cmath>
#include <utility>
#include <thread>
#include <tuple>
#include <mutex>
#include <condition_variable>

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

// =============================================================================
// Port names, configs, and definitions used by this process.
create_port_trait( initializations, object_track_set, "Input initializations" );
create_port_trait( short_term_tracks, object_track_set, "Input ST tracks" );
create_port_trait( short_term_timestamp, timestamp, "Input ST tracks timestamp" );
create_port_trait( mid_term_tracks, object_track_set, "Input MT tracks" );
create_port_trait( mid_term_timestamp, timestamp, "Input MT tracks timestamp" );
create_port_trait( long_term_tracks, object_track_set, "Input LT tracks" );
create_port_trait( long_term_timestamp, timestamp, "Input LT tracks timestamp" );

create_port_trait( short_term_initializations, object_track_set, "Init signals" );
create_port_trait( mid_term_initializations, object_track_set, "Init signals" );
create_port_trait( long_term_initializations, object_track_set, "Init signals" );

typedef std::tuple< kv::timestamp,
                    kv::image_container_sptr,
                    kv::object_track_set_sptr > image_and_track_tuple_t;
typedef std::list< image_and_track_tuple_t > image_and_track_buffer_t;

typedef std::pair< kv::timestamp,
                   kv::object_track_set_sptr > timestamp_track_pair_t;
typedef std::list< timestamp_track_pair_t > track_buffer_t;

typedef kv::track_id_t track_id_t;

enum track_status{ ALL_TRACKING = 0, SUBSET_TRACKING, NONE_TRACKING };

struct track_info_t
{
  std::vector< kv::track_state_sptr > states;
  track_status status;
  unsigned frames_since_last[3];
};

create_config_trait( synchronize, bool, "true",
  "Expect no frame droppages and wait for outputs from all trackers at each "
  "step. If disabled, downsampling of input trackers is allowed." );
create_config_trait( auto_track_id_start, track_id_t, "10000",
  "If a combination of user and automatically generated tracking is enabled, "
  "this field allows a clear differentiation between the two by starting "
  "automatic track IDs at this value." );


// =============================================================================
// Private implementation class
class track_conductor_process::priv
{
public:
  explicit priv( track_conductor_process* parent );
  ~priv();

  // Configuration settings
  bool m_synchronize;
  track_id_t m_auto_track_id_start;
  unsigned m_mid_term_reinit_thresh;
  unsigned m_long_term_reinit_thresh;

  // Internal thread system
  std::vector< std::thread > threads;
  std::condition_variable update_trigger;

  // Internal buffers
  image_and_track_buffer_t m_standard_inputs;
  std::mutex m_standard_input_mutex;
  track_buffer_t m_short_term_tracks;
  std::mutex m_short_term_mutex;
  bool m_has_short_term_tracker;
  track_buffer_t m_mid_term_tracks;
  std::mutex m_mid_term_mutex;
  bool m_has_mid_term_tracker;
  track_buffer_t m_long_term_tracks;
  std::mutex m_long_term_mutex;
  bool m_has_long_term_tracker;

  // Track management
  std::map< track_id_t, track_info_t > m_active_tracks;

  kv::timestamp m_last_output;
  kv::object_track_set_sptr m_st_corrections;
  kv::object_track_set_sptr m_mt_corrections;
  kv::object_track_set_sptr m_lt_corrections;

  // General frame-level properties
  bool m_is_first;
  bool m_received_complete;
  track_conductor_process* parent;
};


// -----------------------------------------------------------------------------
track_conductor_process::priv
::priv( track_conductor_process* ptr )
  : m_synchronize( true )
  , m_auto_track_id_start( 10000 )
  , m_mid_term_reinit_thresh( 5 )
  , m_long_term_reinit_thresh( 10 )
  , m_has_short_term_tracker( false )
  , m_has_mid_term_tracker( false )
  , m_has_long_term_tracker( false )
  , m_is_first( true )
  , m_received_complete( false )
  , parent( ptr )
{
}


track_conductor_process::priv
::~priv()
{
}


// =============================================================================
track_conductor_process
::track_conductor_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new track_conductor_process::priv( this ) )
{
  set_data_checking_level( check_none );

  make_ports();
  make_config();
}


track_conductor_process
::~track_conductor_process()
{
}


// -----------------------------------------------------------------------------
void
track_conductor_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( image, optional );
  declare_input_port_using_trait( timestamp, required );
  declare_input_port_using_trait( initializations, optional );
  declare_input_port_using_trait( short_term_tracks, optional );
  declare_input_port_using_trait( short_term_timestamp, optional );
  declare_input_port_using_trait( mid_term_tracks, optional );
  declare_input_port_using_trait( mid_term_timestamp, optional );
  declare_input_port_using_trait( long_term_tracks, optional );
  declare_input_port_using_trait( long_term_timestamp, optional );

  // -- outputs --
  declare_output_port_using_trait( image, optional );
  declare_output_port_using_trait( timestamp, optional );
  declare_output_port_using_trait( object_track_set, optional );

  // -- feedback loops --
  declare_output_port_using_trait( short_term_initializations, optional );
  declare_output_port_using_trait( mid_term_initializations, optional );
  declare_output_port_using_trait( long_term_initializations, optional );
}


// -----------------------------------------------------------------------------
void
track_conductor_process
::make_config()
{
  declare_config_using_trait( synchronize );
  declare_config_using_trait( auto_track_id_start );
}


// -----------------------------------------------------------------------------
void
track_conductor_process
::_configure()
{
  d->m_synchronize = config_value_using_trait( synchronize );
  d->m_auto_track_id_start = config_value_using_trait( auto_track_id_start );
}


// -----------------------------------------------------------------------------
void
track_conductor_process
::make_threads()
{
  throw std::runtime_error( "Asynchronous processing not implemented" );
}


// -----------------------------------------------------------------------------
void
track_conductor_process
::wait_for_standard_inputs()
{
  kv::image_container_sptr image;
  kv::timestamp timestamp;
  kv::object_track_set_sptr tracks;

  auto port_info = peek_at_port_using_trait( timestamp );

  if( port_info.datum->type() == sprokit::datum::complete )
  {
    d->m_received_complete = true;
    grab_edge_datum_using_trait( timestamp );

    if( has_input_port_edge_using_trait( image ) )
    {
      grab_edge_datum_using_trait( image );
    }
    if( has_input_port_edge_using_trait( initializations ) )
    {
      grab_edge_datum_using_trait( initializations );
    }
  }
  else
  {
    timestamp = grab_from_port_using_trait( timestamp );

    if( has_input_port_edge_using_trait( image ) )
    {
      image = grab_from_port_using_trait( image );
    }

    if( has_input_port_edge_using_trait( initializations ) )
    {
      tracks = grab_from_port_using_trait( initializations );
    }

    if( d->m_synchronize )
    {
      d->m_standard_inputs.push_back(
        std::make_tuple( timestamp, image, tracks ) );
    }
    else
    {
      std::lock_guard< std::mutex >( d->m_standard_input_mutex );

      d->m_standard_inputs.push_back(
        std::make_tuple( timestamp, image, tracks ) );
    }
  }  
}


// -----------------------------------------------------------------------------
#define DECLARE_INPUT_FUNCTION( TRACKER )                                    \
void                                                                         \
track_conductor_process                                                      \
::wait_for_ ## TRACKER ## _inputs()                                          \
{                                                                            \
  kv::timestamp timestamp;                                                   \
  kv::object_track_set_sptr tracks;                                          \
  auto port_info = peek_at_port_using_trait( TRACKER ## _timestamp );        \
                                                                             \
  if( port_info.datum->type() == sprokit::datum::complete )                  \
  {                                                                          \
    d->m_received_complete = true;                                           \
                                                                             \
    grab_edge_datum_using_trait( TRACKER ## _timestamp );                    \
    grab_edge_datum_using_trait( TRACKER ## _tracks );                       \
  }                                                                          \
  else                                                                       \
  {                                                                          \
    timestamp = grab_from_port_using_trait( TRACKER ## _timestamp );         \
    tracks = grab_from_port_using_trait( TRACKER ## _tracks );               \
                                                                             \
    if( d->m_synchronize )                                                   \
    {                                                                        \
      d->m_ ## TRACKER ## _tracks.push_back(                                 \
        std::make_pair( timestamp, tracks ) );                               \
    }                                                                        \
    else                                                                     \
    {                                                                        \
      std::lock_guard< std::mutex >( d->m_standard_input_mutex );            \
      d->m_ ## TRACKER ## _tracks.push_back(                                 \
        std::make_pair( timestamp, tracks ) );                               \
    }                                                                        \
  }                                                                          \
}

DECLARE_INPUT_FUNCTION( short_term )
DECLARE_INPUT_FUNCTION( mid_term )
DECLARE_INPUT_FUNCTION( long_term )

// -----------------------------------------------------------------------------
void
track_conductor_process
::_step()
{
  if( d->m_is_first )
  {
    d->m_has_short_term_tracker =
      has_input_port_edge_using_trait( short_term_tracks );
    d->m_has_mid_term_tracker =
      has_input_port_edge_using_trait( mid_term_tracks );
    d->m_has_long_term_tracker =
      has_input_port_edge_using_trait( long_term_tracks );

    if( !d->m_synchronize )
    {
      make_threads();
    }

    d->m_is_first = false;
  }

  if( d->m_synchronize )
  {
    sync_step();
  }
  else
  {
    async_step();
  }
}


// -----------------------------------------------------------------------------
kv::object_track_set_sptr
merge_init_signals( const kv::object_track_set_sptr priority,
                    const kv::object_track_set_sptr secondary )
{
  if( !priority && !secondary )
  {
    return kv::object_track_set_sptr();
  }
  else if( !priority )
  {
    return secondary;
  }
  else if( !secondary )
  {
    return priority;
  }

  std::map< track_id_t, kv::track_sptr > to_output;

  for( auto track : priority->tracks() )
  {
    if( !track )
    {
      continue;
    }

    auto it = to_output.find( track->id() );

    if( it == to_output.end() ||
        track->last_frame() >= it->second->last_frame() )
    {
      to_output[ track->id() ] = track;
    }
  }

  for( auto track : secondary->tracks() )
  {
    if( !track )
    {
      continue;
    }

    auto it = to_output.find( track->id() );

    if( it == to_output.end() ||
        track->last_frame() > it->second->last_frame() )
    {
      to_output[ track->id() ] = track;
    }
  }

  std::vector< kv::track_sptr > track_vec;

  for( auto elem : to_output )
  {
    track_vec.push_back( elem.second );
  }

  return kv::object_track_set_sptr( new kv::object_track_set( track_vec ) );
}


// -----------------------------------------------------------------------------
void
track_conductor_process
::sync_step()
{
  wait_for_standard_inputs();

  if( d->m_received_complete )
  {
    mark_process_as_complete();

    const sprokit::datum_t dat = sprokit::datum::complete_datum();

    push_datum_to_port_using_trait( image, dat );
    push_datum_to_port_using_trait( timestamp, dat );
    push_datum_to_port_using_trait( object_track_set, dat );

    push_datum_to_port_using_trait( short_term_initializations, dat );
    push_datum_to_port_using_trait( mid_term_initializations, dat );
    push_datum_to_port_using_trait( long_term_initializations, dat );
    return;
  }

  // Drive all connected tracks
  const image_and_track_tuple_t& inputs = d->m_standard_inputs.back();

  const kv::timestamp timestamp = std::get<0>( inputs );
  const kv::image_container_sptr image = std::get<1>( inputs );

  if( d->m_has_short_term_tracker )
  {
    auto sti = merge_init_signals( std::get<2>( inputs ), d->m_st_corrections );
    push_to_port_using_trait( short_term_initializations, sti );
  }
  if( d->m_has_mid_term_tracker )
  {
    auto mti = merge_init_signals( std::get<2>( inputs ), d->m_mt_corrections );
    push_to_port_using_trait( mid_term_initializations, mti );
  }
  if( d->m_has_long_term_tracker )
  {
    auto lti = merge_init_signals( std::get<2>( inputs ), d->m_lt_corrections );
    push_to_port_using_trait( long_term_initializations, lti );
  }

  if( d->m_has_short_term_tracker )
  {
    wait_for_short_term_inputs();
  }
  if( d->m_has_mid_term_tracker )
  {
    wait_for_mid_term_inputs();
  }
  if( d->m_has_long_term_tracker )
  {
    wait_for_long_term_inputs();
  }

  // Generate aggregate tracks for current frame
  struct tri_track_t
  {
    kv::track_sptr st;
    kv::track_sptr mt;
    kv::track_sptr lt;
  };

  std::map< track_id_t, tri_track_t > computed_tracks;

  if( d->m_has_short_term_tracker )
  {
    for( auto track : d->m_short_term_tracks.back().second->tracks() )
    {
      computed_tracks[ track->id() ].st = track;
    }
  }
  if( d->m_has_mid_term_tracker )
  {
    for( auto track : d->m_mid_term_tracks.back().second->tracks() )
    {
      computed_tracks[ track->id() ].mt = track;
    }
  }
  if( d->m_has_long_term_tracker )
  {
    for( auto track : d->m_long_term_tracks.back().second->tracks() )
    {
      computed_tracks[ track->id() ].lt = track;
    }
  }

  // Perform filtering actions (corrections, restarts)
  std::vector< kv::track_sptr > st_corrections;
  std::vector< kv::track_sptr > mt_corrections;
  std::vector< kv::track_sptr > lt_corrections;

  // - update existing tracks
  for( auto track_it : d->m_active_tracks )
  {
    const track_id_t id = track_it.first;

    auto computed_itr = computed_tracks.find( id );
    track_info_t& track_info = d->m_active_tracks[id];

    const bool received_any = ( computed_itr != computed_tracks.end() );

    const tri_track_t& computed = ( received_any ?
      computed_itr->second : tri_track_t() );

    const bool has_st = ( received_any && computed.st &&
      computed.st->last_frame() == timestamp.get_frame() );
    const bool has_mt = ( received_any && computed.mt &&
      computed.mt->last_frame() == timestamp.get_frame() );
    const bool has_lt = ( received_any && computed.lt &&
      computed.lt->last_frame() == timestamp.get_frame() );

    // Update counter states to help with transitions
    track_info.frames_since_last[0] = ( has_st ?
      0 : track_info.frames_since_last[0] + 1 );
    track_info.frames_since_last[1] = ( has_mt ?
      0 : track_info.frames_since_last[1] + 1 );
    track_info.frames_since_last[2] = ( has_lt ?
      0 : track_info.frames_since_last[2] + 1 );

    if( has_st && has_mt && has_lt )
    {
      track_info.status = ALL_TRACKING;
    }
    else if( !has_st && !has_mt && !has_lt )
    {
      track_info.status = NONE_TRACKING;
    }
    else
    {
      track_info.status = SUBSET_TRACKING;
    }

    // Update track states for current frame
    if( has_st && has_mt && has_lt )
    {
      track_info.states.push_back( computed.st->back() );
    }
    else if( has_mt && has_lt )
    {
      track_info.states.push_back( computed.lt->back() );

      st_corrections.push_back( computed.lt );
    }
    else if( has_mt && has_st )
    {
      track_info.states.push_back( computed.st->back() );
    }
    else if( has_lt && has_st )
    {
      track_info.states.push_back( computed.st->back() );

      st_corrections.push_back( computed.mt );
    }
    else if( has_lt )
    {
      track_info.states.push_back( computed.mt->back() );

      st_corrections.push_back( computed.lt );
    }
    else if( has_mt )
    {
      track_info.states.push_back( computed.mt->back() );

      st_corrections.push_back( computed.mt );

      if( track_info.frames_since_last[2] > d->m_long_term_reinit_thresh )
      {
        lt_corrections.push_back( computed.st );
      }
    }
    else if( has_st )
    {
      track_info.states.push_back( computed.st->back() );

      if( track_info.frames_since_last[1] > d->m_mid_term_reinit_thresh )
      {
        mt_corrections.push_back( computed.st );
      }
      if( track_info.frames_since_last[2] > d->m_long_term_reinit_thresh )
      {
        lt_corrections.push_back( computed.st );
      }
    }
  }

  // - handle new tracks
  for( auto tri_itr : computed_tracks )
  {
    if( d->m_active_tracks.find( tri_itr.first ) == d->m_active_tracks.end() )
    {
      track_info_t new_track_info;

      new_track_info.states.push_back ( tri_itr.second.st->back() ?
        tri_itr.second.st->back() : tri_itr.second.mt->back() );
 
      new_track_info.status = ALL_TRACKING;
      std::fill( std::begin(new_track_info.frames_since_last),
                 std::end(new_track_info.frames_since_last), 0 );

      d->m_active_tracks[ tri_itr.first ] = new_track_info;
    }
  }

  d->m_st_corrections =
    kv::object_track_set_sptr( new kv::object_track_set( st_corrections ) );
  d->m_mt_corrections =
    kv::object_track_set_sptr( new kv::object_track_set( mt_corrections ) );
  d->m_lt_corrections =
    kv::object_track_set_sptr( new kv::object_track_set( lt_corrections ) );

  // Send outputs to all downstream nodes
  std::vector< kv::track_sptr > ot;

  for( auto it : d->m_active_tracks )
  {
    ot.push_back( kv::track_sptr( kv::track::create() ) );
    ot.back()->set_id( it.first );

    for( auto state : it.second.states )
    {
      ot.back()->append( state );
    }
  }

  kv::object_track_set_sptr output( new kv::object_track_set( ot ) );

  push_to_port_using_trait( timestamp, timestamp );
  push_to_port_using_trait( image, image );
  push_to_port_using_trait( object_track_set, output );

  d->m_standard_inputs.pop_back();

  if( d->m_has_short_term_tracker )
  {
    d->m_short_term_tracks.pop_back();
  }
  if( d->m_has_mid_term_tracker )
  {
    d->m_mid_term_tracks.pop_back();
  }
  if( d->m_has_long_term_tracker )
  {
    d->m_long_term_tracks.pop_back();
  }
}


// -----------------------------------------------------------------------------
void
track_conductor_process
::async_step()
{
  throw std::runtime_error( "Asynchronous processing not implemented" );
}


} // end namespace core

} // end namespace viame
