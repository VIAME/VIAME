// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "detected_object_type.h"
#include "detected_object.h"
#include "bounding_box.h"
#include "convert_protobuf.h"

#include <vital/types/detected_object.h>
#include <vital/types/protobuf/detected_object.pb.h>
#include <vital/exceptions.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// ----------------------------------------------------------------------------
detected_object::
detected_object()
{
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}

detected_object::
~detected_object()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
detected_object::
serialize( const vital::any& element )
{
  kwiver::vital::detected_object det_object =
    kwiver::vital::any_cast< kwiver::vital::detected_object > ( element );

  std::ostringstream msg;
  msg << "detected_object "; // add type tag

  kwiver::protobuf::detected_object proto_det_object;
  convert_protobuf( det_object, proto_det_object );

  if ( ! proto_det_object.SerializeToOstream( &msg ) )
  {
    VITAL_THROW( kwiver::vital::serialization_exception,
                 "Error serializing detected_object from protobuf" );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any detected_object::
deserialize( const std::string& message )
{
  std::istringstream msg( message );
  auto det_object_ptr = std::make_shared< kwiver::vital::detected_object >(
    kwiver::vital::bounding_box_d { 0, 0, 0, 0 } );

  std::string tag;
  msg >> tag;
  msg.get();  // Eat delimiter

  if (tag != "detected_object" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"detected_object\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    // define our protobuf
    kwiver::protobuf::detected_object proto_det_object;
    if ( ! proto_det_object.ParseFromIstream( &msg ) )
    {
      VITAL_THROW( kwiver::vital::serialization_exception,
                   "Error deserializing detected_object from protobuf" );
    }

    convert_protobuf( proto_det_object, *det_object_ptr );
  }

  return kwiver::vital::any( det_object_ptr );
}

} } } } // end namespace
