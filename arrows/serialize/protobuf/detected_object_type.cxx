// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "detected_object_type.h"
#include "convert_protobuf.h"

#include <vital/types/detected_object_type.h>
#include <vital/types/protobuf/detected_object_type.pb.h>
#include <vital/exceptions.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// ----------------------------------------------------------------------------
detected_object_type::
detected_object_type()
{
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}

detected_object_type::
~detected_object_type()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
detected_object_type::
serialize( const vital::any& element )
{
  kwiver::vital::detected_object_type dot =
    kwiver::vital::any_cast< kwiver::vital::detected_object_type > ( element );

  std::ostringstream msg;
  msg << "detected_object_type "; // add type tag

  kwiver::protobuf::detected_object_type proto_dot;
  convert_protobuf( dot, proto_dot );

  if ( ! proto_dot.SerializeToOstream( &msg ) )
  {
    VITAL_THROW( kwiver::vital::serialization_exception,
                 "Error serializing detected_object_type from protobuf" );
  }
  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any detected_object_type::
deserialize( const std::string& message )
{
  kwiver::vital::detected_object_type dot;
  std::istringstream msg( message );

  std::string tag;
  msg >> tag;
  msg.get();  // Eat delimiter

  if (tag != "detected_object_type" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"detected_object_type\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    // define our protobuf
    kwiver::protobuf::detected_object_type proto_dot;
    if ( ! proto_dot.ParseFromIstream( &msg ) )
    {
      VITAL_THROW( kwiver::vital::serialization_exception,
                   "Error deserializing detected_type from protobuf" );
    }

    convert_protobuf( proto_dot, dot );
  }

  return kwiver::vital::any(dot);
}

} } } } // end namespace
