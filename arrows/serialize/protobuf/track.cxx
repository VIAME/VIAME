// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "track.h"
#include "convert_protobuf.h"

#include <vital/types/track.h>
#include <vital/types/protobuf/track.pb.h>
#include <vital/exceptions.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// ----------------------------------------------------------------------------
track::track()
{
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}

track::~track()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
track::
serialize( const vital::any& element )
{
  kwiver::vital::track_sptr trk_sptr = 
    kwiver::vital::any_cast< kwiver::vital::track_sptr > ( element );

  std::ostringstream msg;
  msg << "track "; // add type tag

  kwiver::protobuf::track proto_trk;
  convert_protobuf( trk_sptr, proto_trk );

  if ( ! proto_trk.SerializeToOstream( &msg ) )
  {
    VITAL_THROW( kwiver::vital::serialization_exception,
                 "Error serializing track from protobuf" );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any track::
deserialize( const std::string& message )
{
  auto trk_sptr = kwiver::vital::track::create();
  std::istringstream msg( message );

  std::string tag;
  msg >> tag;
  msg.get();  // Eat delimiter

  if (tag != "track" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"track\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    // define our protobuf
    kwiver::protobuf::track proto_trk;
    if ( ! proto_trk.ParseFromIstream( &msg ) )
    {
      VITAL_THROW( kwiver::vital::serialization_exception,
                   "Error deserializing track from protobuf" );
    }

    convert_protobuf( proto_trk, trk_sptr );
  }

  return kwiver::vital::any( trk_sptr );
}

} } } } // end namespace
