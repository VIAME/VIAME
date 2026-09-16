// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "merge_track_sets_process.h"

#include <viame/core_types/object_track_set.h>
#include <viame/algorithm_framework/util/string.h>
#include <viame/core_types/vital_types.h>

#include <viame/pipeline_framework/type_traits.h>

#include <viame/pipeline_framework/process_exception.h>

namespace viame {

// -----------------------------------------------------------------------------
// Private implementation class
class merge_track_sets_process::priv
{
public:
  priv() : highest_id( 0 ) {}
  ~priv() {}


  std::set< std::string > p_port_list;

  viame::track_id_t highest_id;
  std::set< viame::track_id_t > used_ids;
  std::vector< std::map< viame::track_id_t, viame::track_id_t > > id_remapping;

  void add_tracks_to_set(
    viame::track_set_sptr input, unsigned index,
    viame::track_set_sptr& output );
};

void
merge_track_sets_process::priv
::add_tracks_to_set(
  viame::track_set_sptr input, unsigned index,
  viame::track_set_sptr& output )
{
  if( !input )
  {
    return;
  }


  std::map< viame::track_id_t,
    viame::track_id_t >& mappings = id_remapping[ index ];

  for( auto track_ptr : input->tracks() )
  {
    if( !track_ptr )
    {
      continue;
    }


    const viame::track_id_t id = track_ptr->id();
    viame::track_id_t mapped_id;
    auto element = mappings.find( id );

    if( element != mappings.end() )
    {
      mapped_id = element->second;
    }
    else
    {
      if( used_ids.find( id ) == used_ids.end() )
      {
        mapped_id = id;
        highest_id = std::max( highest_id, id );
      }
      else
      {
        highest_id = highest_id + 1;
        mapped_id = highest_id;
      }

      used_ids.insert( mapped_id );
      mappings[ id ] = mapped_id;
    }

    track_ptr->set_id( mapped_id );
    output->insert( track_ptr );
  }
}

// ============================================================================

merge_track_sets_process
::merge_track_sets_process( viame::config_block_sptr const& config )
  : process( config ),
    d( new merge_track_sets_process::priv )
{
  make_ports();
  make_config();
}

merge_track_sets_process
::~merge_track_sets_process()
{}

// ----------------------------------------------------------------------------
void
merge_track_sets_process
::_configure()
{}

// ----------------------------------------------------------------------------
void
merge_track_sets_process
::_step()
{
  std::vector< viame::track_set_sptr > track_list;

  for( const auto port_name : d->p_port_list )
  {
    viame::track_set_sptr track_sptr =
      grab_from_port_as< viame::object_track_set_sptr >( port_name );

    track_list.push_back( track_sptr );
  }

  if( track_list.size() > d->id_remapping.size() )
  {
    d->id_remapping.resize( track_list.size() );
  }


  // Merge tracks sequentially
  viame::track_set_sptr output = std::make_shared< viame::object_track_set >();

  if( track_list.empty() )
  {
    LOG_WARN( logger(), "No input tracks provided" );
  }

  for( unsigned i = 0; i < track_list.size(); ++i )
  {
    d->add_tracks_to_set( track_list[ i ], i, output );
  }

  // Return by value
  push_to_port_using_trait(
    object_track_set,
    std::dynamic_pointer_cast< viame::object_track_set >( output ) );
}

// ----------------------------------------------------------------------------
void
merge_track_sets_process
::make_ports()
{
  // Set up for required ports
  viame::pipeline::process::port_flags_t optional;
  viame::pipeline::process::port_flags_t required;
  required.insert( flag_required );

  // -- output --
  declare_output_port_using_trait( object_track_set, required );
}

// ----------------------------------------------------------------------------
void
merge_track_sets_process
::make_config()
{}

// ============================================================================
viame::pipeline::process::port_info_t
merge_track_sets_process
::_input_port_info( port_t const& port_name )
{
  LOG_TRACE( logger(), "Processing input port info: \"" << port_name << "\"" );

  // Just create an input port to read detections from
  if( !viame::starts_with( port_name, "_" ) )
  {
    // Check for unique port name
    if( d->p_port_list.count( port_name ) == 0 )
    {
      port_flags_t required;
      required.insert( flag_required );

      // Create input port
      declare_input_port(
        port_name,                                // port name
        object_track_set_port_trait::type_name,   // port type
        required,                                 // port flags
        "track input" );

      d->p_port_list.insert( port_name );
    }
  }

  // call base class implementation
  return process::_input_port_info( port_name );
}

} // namespace viame
