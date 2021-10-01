/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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
 * \brief Unwrap the object detections from object tracks.
 */

#include "unwrap_detections_process.h"

#include <vital/vital_types.h>

#include <kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

namespace kwiver
{

//----------------------------------------------------------------
// Private implementation class
class unwrap_detections_process::priv
{
public:
  priv();
  ~priv();

  vital::frame_id_t m_current_idx;
};


// ===============================================================================

unwrap_detections_process
::unwrap_detections_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new unwrap_detections_process::priv )
{
  make_ports();
  make_config();
}


unwrap_detections_process
::~unwrap_detections_process()
{
}


// -------------------------------------------------------------------------------
void
unwrap_detections_process
::_configure()
{
}


// -------------------------------------------------------------------------------
void
unwrap_detections_process
::_step()
{
  auto object_tracks = grab_from_port_using_trait( object_track_set );
  auto detected_objects = std::make_shared< kwiver::vital::detected_object_set >();

  if( object_tracks )
  {
    for( auto& trk : object_tracks->tracks() )
    {
      for( auto& state : *trk )
      {
        auto obj_state =
          std::static_pointer_cast< kwiver::vital::object_track_state >( state );

        if( state->frame() == d->m_current_idx )
        {
          detected_objects->add( obj_state->detection() );
        }
      }
    }
  }

  push_to_port_using_trait( detected_object_set, detected_objects );

  d->m_current_idx++;
}


// -------------------------------------------------------------------------------
void
unwrap_detections_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( object_track_set, required );

  // -- output --
  declare_output_port_using_trait( detected_object_set, required );
}


// -------------------------------------------------------------------------------
void
unwrap_detections_process
::make_config()
{
}


// ===============================================================================
unwrap_detections_process::priv
::priv()
  : m_current_idx( 0 )
{
}


unwrap_detections_process::priv
::~priv()
{
}

} // end namespace
