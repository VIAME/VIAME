/*ckwg +29
 * Copyright 2025 by Kitware, Inc.
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
 * \brief Run manual measurement on input tracks
 */

#include "manual_measurement_process.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/object_track_set.h>
#include <vital/util/string.h>

#include <sprokit/processes/kwiver_type_traits.h>

#include <string>
#include <map>

namespace kv = kwiver::vital;

namespace viame
{

namespace core
{

create_config_trait( calibration_file, std::string, "",
  "Input filename for the calibration file to use" );

create_port_trait( object_track_set1, object_track_set,
  "The stereo filtered object tracks1.")
create_port_trait( object_track_set2, object_track_set,
  "The stereo filtered object tracks2.")

// =============================================================================
// Private implementation class
class manual_measurement_process::priv
{
public:
  explicit priv( manual_measurement_process* parent );
  ~priv();

  // Configuration settings
  std::string m_calibration_file;

  // Other variables
  std::set< std::string > p_port_list;
  unsigned m_frame_counter;
  manual_measurement_process* parent;
};


// -----------------------------------------------------------------------------
manual_measurement_process::priv
::priv( manual_measurement_process* ptr )
  : m_calibration_file( "" )
  , m_frame_counter( 0 )
  , parent( ptr )
{
}


manual_measurement_process::priv
::~priv()
{
}

// =============================================================================
manual_measurement_process
::manual_measurement_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new manual_measurement_process::priv( this ) )
{
  this->set_data_checking_level( check_none );

  make_ports();
  make_config();
}


manual_measurement_process
::~manual_measurement_process()
{
}


// -----------------------------------------------------------------------------
void
manual_measurement_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( timestamp, optional );

  // -- outputs --
  declare_output_port_using_trait( object_track_set1, required );
  declare_output_port_using_trait( object_track_set2, required );
  declare_output_port_using_trait( timestamp, optional );
}

// -----------------------------------------------------------------------------
void
manual_measurement_process
::make_config()
{
  declare_config_using_trait( calibration_file );
}

// -----------------------------------------------------------------------------
void
manual_measurement_process
::_configure()
{
  d->m_calibration_file = config_value_using_trait( calibration_file );
}

// ----------------------------------------------------------------------------
void
manual_measurement_process
::_init()
{
  this->set_data_checking_level( check_valid );
}

// ----------------------------------------------------------------------------
void
manual_measurement_process
::input_port_undefined( port_t const& port_name )
{
  LOG_TRACE( logger(), "Processing undefined input port: \"" << port_name << "\"" );

  // Just create an input port to read detections from
  if( !kv::starts_with( port_name, "_" ) )
  {
    // Check for unique port name
    if( d->p_port_list.count( port_name ) == 0 )
    {
      port_flags_t required;
      required.insert( flag_required );

      // Create input port
      declare_input_port(
        port_name,                                 // port name
        object_track_set_port_trait::type_name,    // port type
        required,                                  // port flags
        "object track set input" );

      d->p_port_list.insert( port_name );
    }
  }
}

// -----------------------------------------------------------------------------
void
manual_measurement_process
::_step()
{
  std::vector< kv::object_track_set_sptr > inputs;
  kv::object_track_set_sptr output;
  kv::timestamp ts;

  for( auto const& port_name : d->p_port_list )
  {
    if( port_name != "timestamp" )
    {
      inputs.push_back(
        grab_from_port_as< kv::object_track_set_sptr >( port_name ) );
    }
    else
    {
      ts = grab_from_port_using_trait( timestamp );
    }
  }

  kv::frame_id_t cur_frame_id = ( ts.has_valid_frame() ?
                                  ts.get_frame() :
                                  d->m_frame_counter );

  d->m_frame_counter++;

  if( inputs.size() != 2 )
  {
    const std::string err = "Currently only 2 camera inputs are supported";
    LOG_ERROR( logger(), err );
    throw std::runtime_error( err );
  }

  // Identify all input detections across all track sets on the current frame
  typedef std::vector< std::map< kv::track_id_t, kv::detected_object_sptr > > map_t;
  map_t dets( inputs.size() );

  for( unsigned i = 0; i < inputs.size(); ++i )
  {
    for( auto& trk : inputs[i]->tracks() )
    {
      for( auto& state : *trk )
      {
        auto obj_state =
          std::static_pointer_cast< kwiver::vital::object_track_state >( state );

        if( state->frame() == cur_frame_id )
        {
          dets[i][trk->id()] = obj_state->detection();
        }
      }
    }
  }

  // Identify which detections are matched on the current frame
  std::vector< kv::track_id_t > common_ids;

  for( auto itr : dets[0] )
  {
    bool found_match = false;

    for( unsigned i = 1; i < inputs.size(); ++i )
    {
      if( dets[i].find( itr.first ) != dets[i].end() )
      {
        found_match = true;
        common_ids.push_back( itr.first );
        break;
      }
    }

    if( found_match )
    {
      LOG_INFO( logger(), "Found match for track ID " + std::to_string( itr.first ) );
    }
    else
    {
      LOG_INFO( logger(), "No match for track ID " + std::to_string( itr.first ) );
    }
  }

  // Run measurement on matched detections
  // TODO

  // Push outputs
  push_to_port_using_trait( object_track_set1, inputs[0] );
  push_to_port_using_trait( object_track_set2, inputs[1] );
  push_to_port_using_trait( timestamp, ts );
}

} // end namespace core

} // end namespace viame
