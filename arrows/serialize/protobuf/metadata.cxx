// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "metadata.h"
#include "convert_protobuf.h"

#include <vital/types/metadata.h>
#include <vital/types/protobuf/metadata.pb.h>

#include <typeinfo>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// ----------------------------------------------------------------------------
metadata::
metadata()
{
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}

metadata::
~metadata()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
metadata::
serialize( const vital::any& element )
{
  kwiver::vital::metadata_vector mvec =
    kwiver::vital::any_cast< kwiver::vital::metadata_vector > ( element );

  std::ostringstream msg;
  msg << "metadata "; // add type tag

  kwiver::protobuf::metadata_vector proto_mvec;
  convert_protobuf( mvec, proto_mvec );

  if ( ! proto_mvec.SerializeToOstream( &msg ) )
  {
    LOG_ERROR( logger(), "proto_mvec.SerializeToOStream failed" );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any metadata::
deserialize( const std::string& message )
{
  kwiver::vital::metadata_vector mvec;
  std::istringstream msg( message );
  std::string tag;
  msg >> tag;
  msg.get();  // Eat the delimiter

  if (tag != "metadata" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"metadata\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    // define our protobuf
    kwiver::protobuf::metadata_vector proto_mvec;
    if ( ! proto_mvec.ParseFromIstream( &msg ) )
    {
      LOG_ERROR( logger(), "Incoming protobuf stream did not parse correctly. ParseFromIstream failed." );
    }

    convert_protobuf( proto_mvec, mvec );
  }

  return kwiver::vital::any(mvec);
}

} } } } // end namespace
