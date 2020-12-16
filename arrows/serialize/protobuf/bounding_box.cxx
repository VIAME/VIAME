// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "bounding_box.h"
#include "convert_protobuf.h"

#include <vital/types/bounding_box.h>
#include <vital/types/protobuf/bounding_box.pb.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// ----------------------------------------------------------------------------
bounding_box::
bounding_box()
{
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}

bounding_box::
~bounding_box()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
bounding_box::
serialize( const vital::any& element )
{
  kwiver::vital::bounding_box_d bbox =
    kwiver::vital::any_cast< kwiver::vital::bounding_box_d > ( element );

  std::ostringstream msg;
  msg << "bounding_box "; // add type tag

  kwiver::protobuf::bounding_box proto_bbox;
  convert_protobuf( bbox, proto_bbox );

  if ( ! proto_bbox.SerializeToOstream( &msg ) )
  {
    LOG_ERROR( logger(), "proto_bbox.SerializeToOStream failed" );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any bounding_box::
deserialize( const std::string& message )
{
  kwiver::vital::bounding_box_d bbox{ 0, 0, 0, 0 };

  std::istringstream msg( message );
  std::string tag;
  msg >> tag;
  msg.get();  // Eat the delimiter

  if (tag != "bounding_box" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"bounding_box\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    // define our protobuf
    kwiver::protobuf::bounding_box proto_bbox;
    if ( ! proto_bbox.ParseFromIstream( &msg ) )
    {
      LOG_ERROR( logger(), "Incoming protobuf stream did not parse correctly. ParseFromIstream failed." );
    }

    convert_protobuf( proto_bbox, bbox );
  }

  return kwiver::vital::any(bbox);
}

} } } } // end namespace
