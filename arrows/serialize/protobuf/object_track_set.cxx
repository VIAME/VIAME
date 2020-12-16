// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "object_track_set.h"
#include "convert_protobuf.h"

#include <vital/types/object_track_set.h>
#include <vital/types/protobuf/object_track_set.pb.h>
#include <vital/exceptions.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// ----------------------------------------------------------------------------
object_track_set::object_track_set()
{
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}

object_track_set::~object_track_set()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
object_track_set::
serialize( const vital::any& element )
{
  kwiver::vital::object_track_set_sptr obj_trk_set_sptr = 
    kwiver::vital::any_cast< kwiver::vital::object_track_set_sptr > ( element );

  std::ostringstream msg;
  msg << "object_track_set "; // add type tag

  kwiver::protobuf::object_track_set proto_obj_trk_set;
  convert_protobuf( obj_trk_set_sptr, proto_obj_trk_set );

  if ( ! proto_obj_trk_set.SerializeToOstream( &msg ) )
  {
    VITAL_THROW( kwiver::vital::serialization_exception,
                 "Error serializing track from protobuf" );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any object_track_set::
deserialize( const std::string& message )
{
  auto obj_trk_set_sptr = std::make_shared< kwiver::vital::object_track_set >();
  std::istringstream msg( message );

  std::string tag;
  msg >> tag;
  msg.get();  // Eat delimiter

  if (tag != "object_track_set" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"object track set\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    // define our protobuf
    kwiver::protobuf::object_track_set proto_obj_trk_set;
    if ( ! proto_obj_trk_set.ParseFromIstream( &msg ) )
    {
      VITAL_THROW( kwiver::vital::serialization_exception,
                   "Error deserializing track from protobuf" );
    }

    convert_protobuf( proto_obj_trk_set, obj_trk_set_sptr );
  }

  return kwiver::vital::any( obj_trk_set_sptr );
}

} } } } // end namespace
