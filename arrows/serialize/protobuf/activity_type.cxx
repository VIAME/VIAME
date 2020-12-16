// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "activity_type.h"
#include "convert_protobuf.h"

#include <vital/types/activity_type.h>
#include <vital/types/protobuf/activity_type.pb.h>
#include <vital/exceptions.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// ----------------------------------------------------------------------------
activity_type::
activity_type()
{
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}

activity_type::
~activity_type()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
activity_type::
serialize( const vital::any& element )
{
  kwiver::vital::activity_type at =
    kwiver::vital::any_cast< kwiver::vital::activity_type > ( element );

  std::ostringstream msg;
  msg << "activity_type "; // add type tag

  kwiver::protobuf::activity_type proto_at;
  convert_protobuf( at, proto_at );

  if ( ! proto_at.SerializeToOstream( &msg ) )
  {
    VITAL_THROW( kwiver::vital::serialization_exception,
                 "Error serializing detected_type from protobuf" );
  }
  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any activity_type::
deserialize( const std::string& message )
{
  kwiver::vital::activity_type at;
  std::istringstream msg( message );

  std::string tag;
  msg >> tag;
  msg.get();  // Eat delimiter

  if (tag != "activity_type" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"activity_type\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    // define our protobuf
    kwiver::protobuf::activity_type proto_at;
    if ( ! proto_at.ParseFromIstream( &msg ) )
    {
      VITAL_THROW( kwiver::vital::serialization_exception,
                   "Error deserializing detected_type from protobuf" );
    }

    convert_protobuf( proto_at, at );
  }

  return kwiver::vital::any(at);
}

} } } } // end namespace
