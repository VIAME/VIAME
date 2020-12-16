// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "track_state.h"
#include "convert_protobuf.h"

#include "vital/types/track.h"
#include <vital/types/protobuf/track_state.pb.h>
#include <vital/exceptions.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// ----------------------------------------------------------------------------
  track_state::track_state()
  {
    // Verify that the version of the library that we linked against is
    // compatible with the version of the headers we compiled against.
    GOOGLE_PROTOBUF_VERIFY_VERSION;
  }

  track_state::~track_state()
  { }

  
  // ----------------------------------------------------------------------------
  std::shared_ptr< std::string >
  track_state::
  serialize( const vital::any& element )
  {
    kwiver::vital::track_state trk_state =
      kwiver::vital::any_cast< kwiver::vital::track_state > ( element );

    std::ostringstream msg;
    msg << "track_state "; // add type tag

    kwiver::protobuf::track_state proto_trk_state;
    convert_protobuf( trk_state, proto_trk_state );

    if ( ! proto_trk_state.SerializeToOstream( &msg ) )
    {
      VITAL_THROW( kwiver::vital::serialization_exception,
                   "Error serializing track state from protobuf" );
    }

    return std::make_shared< std::string > ( msg.str() );
  }

  // ----------------------------------------------------------------------------
  vital::any track_state::
  deserialize( const std::string& message )
  {
    std::istringstream msg( message );
    kwiver::vital::track_state trk_state( 0 ); 
    std::string tag;
    msg >> tag;
    msg.get();  // Eat delimiter

    if (tag != "track_state" )
    {
      LOG_ERROR( logger(), "Invalid data type tag received. Expected \"track_state\", received \""
                 << tag << "\". Message dropped." );
    }
    else
    {
      // define our protobuf
      kwiver::protobuf::track_state proto_trk_state;
      if ( ! proto_trk_state.ParseFromIstream( &msg ) )
      {
        VITAL_THROW( kwiver::vital::serialization_exception,
                     "Error deserializing Track State from protobuf" );
      }

      convert_protobuf( proto_trk_state,  trk_state );
    }

    return kwiver::vital::any( trk_state );
  }

} } } } // end namespace
