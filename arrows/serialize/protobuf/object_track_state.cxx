// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
