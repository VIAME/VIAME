// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "string.h"

#include "convert_protobuf.h"

#include <vital/types/protobuf/string.pb.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

string::string()
{
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}

string::~string()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string > string::
serialize( const vital::any& element )
{
  const std::string data = kwiver::vital::any_cast< std::string >( element );
  std::ostringstream msg;
  msg << "string ";

  kwiver::protobuf::string proto_string;
  convert_protobuf( data, proto_string );

  if ( ! proto_string.SerializeToOstream( &msg ) )
  {
    LOG_ERROR( logger(), "proto_string.SerializeToOStream failed" );
  }
  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any string::deserialize( const std::string& message )
{
  std::string data;
  std::istringstream msg( message );
  std::string tag;
  msg >> tag;
  msg.get();

  if (tag != "string")
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"string\", received \""
            << tag << "\". Message dropped.");
  }
  else
  {
    kwiver::protobuf::string proto_str;
    if ( !proto_str.ParseFromIstream( &msg ) )
    {
      LOG_ERROR( logger(), "Incoming protobuf stream did not parse correctly");
    }
    convert_protobuf( proto_str, data);
  }
  return kwiver::vital::any(data);
}

} } } }
