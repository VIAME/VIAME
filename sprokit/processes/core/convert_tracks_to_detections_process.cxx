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

#include "convert_tracks_to_detections_process.h"

#include <vital/vital_types.h>

#include <vital/types/timestamp.h>
#include <vital/types/object_track_set.h>

#include <kwiver_type_traits.h>

#include <sprokit/processes/kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

namespace kwiver
{

create_config_trait( frame_ids_only, bool, "false",
  "Only use frame IDs, not entire timestamps, for identifying the current frame." );

//------------------------------------------------------------------------------
// Private implementation class
class convert_tracks_to_detections_process::priv
{
public:
  priv() : frame_ids_only( false ) {}
  ~priv() {}

  bool frame_ids_only;
};


// =============================================================================

convert_tracks_to_detections_process
::convert_tracks_to_detections_process( vital::config_block_sptr const& config )
  : process( config ),
    d( new convert_tracks_to_detections_process::priv )
{
  set_data_checking_level( check_none );

  make_ports();
  make_config();
}


convert_tracks_to_detections_process
::~convert_tracks_to_detections_process()
{
}


// -----------------------------------------------------------------------------
void convert_tracks_to_detections_process
::_configure()
{
  scoped_configure_instrumentation();

  vital::config_block_sptr algo_config = get_config();

  d->frame_ids_only = config_value_using_trait( frame_ids_only );

  process::_configure();
}


// -----------------------------------------------------------------------------
void
convert_tracks_to_detections_process
::_step()
{
  // Check for complete messages
  auto port_info = peek_at_port_using_trait( timestamp );

  if( port_info.datum->type() == sprokit::datum::complete )
  {
    grab_edge_datum_using_trait( timestamp );
    grab_edge_datum_using_trait( object_track_set );
    mark_process_as_complete();

    const sprokit::datum_t dat = sprokit::datum::complete_datum();
    push_datum_to_port_using_trait( detected_object_set, dat );
    return;
  }

  // Retrieve inputs from ports
  vital::timestamp ts = grab_from_port_using_trait( timestamp );
  vital::object_track_set_sptr tracks = grab_from_port_using_trait( object_track_set );

  // Output frame ID
  LOG_DEBUG( logger(), "Processing frame " << ts );

  // Split track set into detections
  std::vector< vital::detected_object_sptr > output;

  if( tracks )
  {
    for( auto trk_ptr : tracks->tracks() )
    {
      if( trk_ptr && !trk_ptr->empty() )
      {
        kwiver::vital::object_track_state* state =
          dynamic_cast< kwiver::vital::object_track_state* >( trk_ptr->back().get() );
  
        if( state &&
            ( ( d->frame_ids_only && state->frame() == ts.get_frame() ) ||
              state->ts() == ts ) )
        {
          output.push_back( state->detection() );
        }
      }
    }
  }

  // Output results
  push_to_port_using_trait( detected_object_set,
    std::make_shared< vital::detected_object_set >( output ) );

  process::_step();
}


// -----------------------------------------------------------------------------
void convert_tracks_to_detections_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( object_track_set, optional );

  // -- output --
  declare_output_port_using_trait( detected_object_set, optional );
}


// -----------------------------------------------------------------------------
void convert_tracks_to_detections_process
::make_config()
{
  declare_config_using_trait( frame_ids_only );
}


} // end namespace
