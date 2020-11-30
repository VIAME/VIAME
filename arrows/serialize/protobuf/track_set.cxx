// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "track_set.h"
#include "convert_protobuf.h"

#include <vital/types/track_set.h>
#include <vital/types/protobuf/track_set.pb.h>
#include <vital/exceptions.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// ----------------------------------------------------------------------------
track_set::track_set()
{
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}

track_set::~track_set()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
track_set::
serialize( const vital::any& element )
{
  kwiver::vital::track_set_sptr trk_set_sptr = 
    kwiver::vital::any_cast< kwiver::vital::track_set_sptr > ( element );

  std::ostringstream msg;
  msg << "track_set "; // add type tag

  kwiver::protobuf::track_set proto_trk_set;
  convert_protobuf( trk_set_sptr, proto_trk_set );

  if ( ! proto_trk_set.SerializeToOstream( &msg ) )
  {
    VITAL_THROW( kwiver::vital::serialization_exception,
                 "Error serializing track from protobuf" );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any track_set::
deserialize( const std::string& message )
{
  auto trk_set_sptr = std::make_shared< kwiver::vital::track_set >();
  std::istringstream msg( message );

  std::string tag;
  msg >> tag;
  msg.get();  // Eat delimiter

  if (tag != "track_set" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"track\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    // define our protobuf
    kwiver::protobuf::track_set proto_trk_set;
    if ( ! proto_trk_set.ParseFromIstream( &msg ) )
    {
      VITAL_THROW( kwiver::vital::serialization_exception,
                   "Error deserializing track from protobuf" );
    }

    convert_protobuf( proto_trk_set, trk_set_sptr );
  }

  return kwiver::vital::any( trk_set_sptr );
}

} } } } // end namespace
