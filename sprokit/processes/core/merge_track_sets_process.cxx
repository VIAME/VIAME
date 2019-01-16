/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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

#include "merge_track_sets_process.h"

#include <vital/vital_types.h>
#include <vital/types/object_track_set.h>
#include <vital/util/string.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace kwiver
{

//-----------------------------------------------------------------------------
// Private implementation class
class merge_track_sets_process::priv
{
public:
  priv() : highest_id( 0 ) {}
  ~priv() {}

  std::set< std::string > p_port_list;

  vital::track_id_t highest_id;
  std::set< vital::track_id_t > used_ids;
  std::vector< std::map< vital::track_id_t, vital::track_id_t > > id_remapping;

  void add_tracks_to_set( vital::track_set_sptr input, unsigned index,
                          vital::track_set_sptr& output );
};


void
merge_track_sets_process::priv
::add_tracks_to_set( vital::track_set_sptr input, unsigned index,
                     vital::track_set_sptr& output )
{
  if( !input )
  {
    return;
  }

  std::map< vital::track_id_t, vital::track_id_t >& mappings = id_remapping[index];

  for( auto track_ptr : input->tracks() )
  {
    if( !track_ptr )
    {
      continue;
    }

    const vital::track_id_t id = track_ptr->id();
    vital::track_id_t mapped_id;
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
::merge_track_sets_process( vital::config_block_sptr const& config )
  : process( config ),
    d( new merge_track_sets_process::priv )
{
  make_ports();
  make_config();
}


merge_track_sets_process
::~merge_track_sets_process()
{
}


// ----------------------------------------------------------------------------
void merge_track_sets_process
::_configure()
{
}


// ----------------------------------------------------------------------------
void
merge_track_sets_process
::_step()
{
  std::vector< vital::track_set_sptr > track_list;

  for( const auto port_name : d->p_port_list )
  {
    vital::track_set_sptr track_sptr =
      grab_from_port_as< vital::object_track_set_sptr >( port_name );

    track_list.push_back( track_sptr );
  }

  if( track_list.size() > d->id_remapping.size() )
  {
    d->id_remapping.resize( track_list.size() );
  }

  // Merge tracks sequentially
  vital::track_set_sptr output = std::make_shared< vital::object_track_set >();

  if( track_list.empty() )
  {
    LOG_WARN( logger(), "No input tracks provided" );
  }

  for( unsigned i = 0; i < track_list.size(); ++i )
  {
    d->add_tracks_to_set( track_list[i], i, output );
  }

  // Return by value
  push_to_port_using_trait( object_track_set,
    std::dynamic_pointer_cast< vital::object_track_set >( output ) );
}


// ----------------------------------------------------------------------------
void merge_track_sets_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- output --
  declare_output_port_using_trait( object_track_set, required );
}


// ----------------------------------------------------------------------------
void merge_track_sets_process
::make_config()
{

}


// ============================================================================
sprokit::process::port_info_t
merge_track_sets_process
::_input_port_info(port_t const& port_name)
{
  LOG_TRACE( logger(), "Processing input port info: \"" << port_name << "\"" );

  // Just create an input port to read detections from
  if( !vital::starts_with( port_name, "_" ) )
  {
    // Check for unique port name
    if( d->p_port_list.count( port_name ) == 0 )
    {
      port_flags_t required;
      required.insert( flag_required );

      // Create input port
      declare_input_port(
          port_name,                              // port name
          object_track_set_port_trait::type_name, // port type
          required,                               // port flags
          "track input" );

      d->p_port_list.insert( port_name );
    }
  }

  // call base class implementation
  return process::_input_port_info( port_name );
}


} // end namespace
