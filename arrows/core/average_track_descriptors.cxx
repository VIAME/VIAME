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

#include "average_track_descriptors.h"

namespace kwiver {
namespace arrows {
namespace core {


// ==================================================================================
class average_track_descriptors::priv
{
public:
  priv()
  {}

  ~priv()
  {}
};


// ==================================================================================
average_track_descriptors
::average_track_descriptors()
  : d( new priv() )
{
}


// ----------------------------------------------------------------------------------
average_track_descriptors
::~average_track_descriptors()
{
}


// ----------------------------------------------------------------------------------
void
average_track_descriptors
::set_configuration( vital::config_block_sptr config )
{
}


// ----------------------------------------------------------------------------------
bool
average_track_descriptors
::check_configuration( vital::config_block_sptr config ) const
{
  return true;
}


// ----------------------------------------------------------------------------------
vital::track_descriptor_set_sptr
average_track_descriptors
::compute( vital::timestamp ts,
           vital::image_container_sptr image_data,
           vital::object_track_set_sptr tracks )
{
  vital::track_descriptor_set_sptr tds( new vital::track_descriptor_set() );

  for( vital::track_sptr track : tracks->tracks() )
  {
    vital::track::history_const_itr it = track->find( ts.get_frame() );
    if( it != track->end() )
    {
      std::shared_ptr< vital::object_track_state > ots =
        std::dynamic_pointer_cast< vital::object_track_state >( *it );
      if( ots )
      {
        vital::track_descriptor_sptr td = vital::track_descriptor::create( "cnn_descriptor" );

        td->set_descriptor( ots->detection->descriptor() );

        tds->push_back( td );
      }
    }
  }

  return tds;
}


// ----------------------------------------------------------------------------------
vital::track_descriptor_set_sptr
average_track_descriptors
::flush()
{
  return vital::track_descriptor_set_sptr();
}


} } }
