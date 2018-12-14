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

#include "object_track_state.h"
#include "convert_protobuf.h"

#include "vital/types/object_track_set.h"
#include <vital/types/protobuf/object_track_state.pb.h>
#include <vital/exceptions.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// ----------------------------------------------------------------------------
  object_track_state::object_track_state()
  {
    // Verify that the version of the library that we linked against is
    // compatible with the version of the headers we compiled against.
    GOOGLE_PROTOBUF_VERIFY_VERSION;
  }

  object_track_state::~object_track_state()
  { }

  
  // ----------------------------------------------------------------------------
  std::shared_ptr< std::string >
  object_track_state::
  serialize( const vital::any& element )
  {
    kwiver::vital::object_track_state obj_trk_state =
      kwiver::vital::any_cast< kwiver::vital::object_track_state >( element );

    std::ostringstream msg;
    msg << "object_track_state "; // add type tag

    kwiver::protobuf::object_track_state proto_obj_trk_state;
    convert_protobuf( obj_trk_state, proto_obj_trk_state );

    if ( ! proto_obj_trk_state.SerializeToOstream( &msg ) )
    {
      VITAL_THROW( kwiver::vital::serialization_exception,
                   "Error serializing track state from protobuf" );
    }

    return std::make_shared< std::string >( msg.str() );
  }

  // ----------------------------------------------------------------------------
  vital::any object_track_state::
  deserialize( const std::string& message )
  {
    std::istringstream msg( message );
    kwiver::vital::object_track_state obj_trk_state; 
    std::string tag;
    msg >> tag;
    msg.get();  // Eat delimiter

    if (tag != "object_track_state" )
    {
      LOG_ERROR( logger(), "Invalid data type tag received. Expected \"object_track_state\", received \""
                 << tag << "\". Message dropped." );
    }
    else
    {
      // define our protobuf
      kwiver::protobuf::object_track_state proto_obj_trk_state;
      if ( ! proto_obj_trk_state.ParseFromIstream( &msg ) )
      {
        VITAL_THROW( kwiver::vital::serialization_exception,
                     "Error deserializing Object Track State from protobuf" );
      }

      convert_protobuf( proto_obj_trk_state,  obj_trk_state);
    }

    return kwiver::vital::any( obj_trk_state );
  }

} } } } // end namespace
