// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "timestamp.h"
#include "convert_protobuf.h"

#include <vital/types/timestamp.h>
#include <vital/types/protobuf/timestamp.pb.h>
#include <vital/exceptions.h>

#include <cstdint>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// ----------------------------------------------------------------------------
timestamp::timestamp()
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}

timestamp::~timestamp()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
timestamp::
serialize( const vital::any& element )
{
  kwiver::vital::timestamp tstamp = kwiver::vital::any_cast< kwiver::vital::timestamp > ( element );
  std::ostringstream msg;
  msg << "timestamp ";
  kwiver::protobuf::timestamp proto_tstamp;

  convert_protobuf( tstamp, proto_tstamp );

  if ( ! proto_tstamp.SerializeToOstream( &msg ) )
  {
    LOG_ERROR( logger(), "proto_timestamp.SerializeToOStream failed" );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any
timestamp::deserialize( const std::string& message )
{
  kwiver::vital::timestamp tstamp;
  std::istringstream msg( message );
  std::string tag;
  msg >> tag;
  msg.get();
  if ( tag != "timestamp" )
  {
    LOG_ERROR( logger(), "Invalid data type tag receiver. Expected timestamp"
               << "received " << tag << ". Message dropped." );
  }
  else
  {
    kwiver::protobuf::timestamp proto_tstamp;
    if ( ! proto_tstamp.ParseFromIstream( &msg ) )
    {
      LOG_ERROR( logger(), "Incoming protobuf stream did not parse correctly."
                 << "ParseFromIstream failed." );
    }

    convert_protobuf( proto_tstamp, tstamp );
  }

  return kwiver::vital::any( tstamp );
}

} } } }
