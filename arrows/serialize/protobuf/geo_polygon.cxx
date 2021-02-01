// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "geo_polygon.h"
#include "convert_protobuf.h"

#include <vital/types/geo_polygon.h>
#include <vital/types/protobuf/geo_polygon.pb.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// ----------------------------------------------------------------------------
geo_polygon::
geo_polygon()
{
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}

geo_polygon::
~geo_polygon()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
geo_polygon::
serialize( const vital::any& element )
{
  kwiver::vital::geo_polygon_d bbox =
    kwiver::vital::any_cast< kwiver::vital::geo_polygon_d > ( element );

  std::ostringstream msg;
  msg << "geo_polygon "; // add type tag

  kwiver::protobuf::geo_polygon proto_bbox;
  convert_protobuf( bbox, proto_bbox );

  if ( ! proto_bbox.SerializeToOstream( &msg ) )
  {
    LOG_ERROR( logger(), "proto_bbox.SerializeToOStream failed" );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any geo_polygon::
deserialize( const std::string& message )
{
  kwiver::vital::geo_polygon_d bbox{ 0, 0, 0, 0 };

  std::istringstream msg( message );
  std::string tag;
  msg >> tag;
  msg.get();  // Eat the delimiter

  if (tag != "geo_polygon" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"geo_polygon\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    // define our protobuf
    kwiver::protobuf::geo_polygon proto_bbox;
    if ( ! proto_bbox.ParseFromIstream( &msg ) )
    {
      LOG_ERROR( logger(), "Incoming protobuf stream did not parse correctly. ParseFromIstream failed." );
    }

    convert_protobuf( proto_bbox, bbox );
  }

  return kwiver::vital::any(bbox);
}

} } } } // end namespace
