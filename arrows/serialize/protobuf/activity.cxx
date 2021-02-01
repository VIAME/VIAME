// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "activity.h"
#include "convert_protobuf.h"

#include <vital/types/activity.h>
#include <vital/types/protobuf/activity.pb.h>
#include <vital/exceptions.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// ----------------------------------------------------------------------------
activity::
activity()
{
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}

activity::
~activity()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
activity::
serialize( const vital::any& element )
{
  kwiver::vital::activity act =
    kwiver::vital::any_cast< kwiver::vital::activity > ( element );

  std::ostringstream msg;
  msg << "activity "; // add type tag

  kwiver::protobuf::activity proto_act;
  convert_protobuf( act, proto_act );

  if ( ! proto_act.SerializeToOstream( &msg ) )
  {
    VITAL_THROW( kwiver::vital::serialization_exception,
                 "Error serializing activity from protobuf" );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any activity::
deserialize( const std::string& message )
{
  kwiver::vital::activity act;
  std::istringstream msg( message );

  std::string tag;
  msg >> tag;
  msg.get();  // Eat delimiter

  if ( tag != "activity" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"activity\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    // define our protobuf
    kwiver::protobuf::activity proto_act;
    if ( ! proto_act.ParseFromIstream( &msg ) )
    {
      VITAL_THROW( kwiver::vital::serialization_exception,
                   "Error deserializing detected_type from protobuf" );
    }

    convert_protobuf( proto_act, act );
  }

  return kwiver::vital::any(act);
}

} } } } // end namespace
