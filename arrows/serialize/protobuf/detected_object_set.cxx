// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "detected_object_set.h"
#include "detected_object.h"
#include "convert_protobuf.h"

#include <vital/types/detected_object_set.h>
#include <vital/types/protobuf/detected_object_set.pb.h>
#include <vital/exceptions.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// ----------------------------------------------------------------------------
detected_object_set::
detected_object_set()
{
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}

detected_object_set::
~detected_object_set()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
detected_object_set::
serialize( const vital::any& element )
{
  kwiver::vital::detected_object_set_sptr dos_sptr =
    kwiver::vital::any_cast< kwiver::vital::detected_object_set_sptr > ( element );

  std::ostringstream msg;
  msg << "detected_object_set "; // add type tag

  kwiver::protobuf::detected_object_set proto_dos;
  convert_protobuf( *dos_sptr, proto_dos );

  if ( ! proto_dos.SerializeToOstream( &msg ) )
  {
    VITAL_THROW( kwiver::vital::serialization_exception,
                 "Error serializing detected_object_set from protobuf" );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any detected_object_set::
deserialize( const std::string& message )
{
  auto dos_sptr = std::make_shared< kwiver::vital::detected_object_set >();
  std::istringstream msg( message );

  std::string tag;
  msg >> tag;
  msg.get();  // Eat delimiter

  if (tag != "detected_object_set" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"detected_object_set\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    // define our protobuf
    kwiver::protobuf::detected_object_set proto_dos;
    if ( ! proto_dos.ParseFromIstream( &msg ) )
    {
      VITAL_THROW( kwiver::vital::serialization_exception,
                   "Error deserializing detected_object_set from protobuf" );
    }

    convert_protobuf( proto_dos, *dos_sptr );
  }

  return kwiver::vital::any(dos_sptr);
}

} } } } // end namespace
