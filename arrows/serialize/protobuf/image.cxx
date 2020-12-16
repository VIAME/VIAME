// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "image.h"
#include "convert_protobuf.h"

#include <vital/types/image_container.h>
#include <vital/types/protobuf/image.pb.h>
#include <vital/exceptions.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// --------------------------------------------------------------------------
image::image()
{
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}

image::~image()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
image::
serialize( const vital::any& element )
{
  kwiver::vital::image_container_sptr img_sptr =
    kwiver::vital::any_cast< kwiver::vital::image_container_sptr > ( element );

  std::ostringstream msg;
  msg << "image ";   // add type tag
  kwiver::protobuf::image proto_img;
  convert_protobuf( img_sptr, proto_img );

  if ( ! proto_img.SerializeToOstream( &msg ) )
  {
    VITAL_THROW( kwiver::vital::serialization_exception,
                 "Error serializing detected_object_set from protobuf" );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any
image::
deserialize( const std::string& message )
{
  kwiver::vital::image_container_sptr img_container_sptr;
  std::istringstream msg( message );

  std::string tag;
  msg >> tag;
  msg.get();    // Eat delimiter

  if ( tag != "image" )
  {
    LOG_ERROR(
      logger(), "Invalid data type tag received. Expected \"image\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    // define our protobuf
    kwiver::protobuf::image proto_img;
    if ( ! proto_img.ParseFromIstream(&msg) )
    {
      VITAL_THROW(kwiver::vital::serialization_exception,
                  "Error deserializing image_container from protobuf");
    }

    convert_protobuf(proto_img, img_container_sptr);
  }

  return kwiver::vital::any( img_container_sptr );
}

} } } }
