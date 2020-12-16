// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "timestamp.h"
#include "load_save.h"

#include <vital/types/timestamp.h>
#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>

#include <sstream>
#include <cstdint>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

// ----------------------------------------------------------------------------
timestamp::timestamp()
{ }

timestamp::~timestamp()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
timestamp
::serialize( const vital::any& element )
{
  kwiver::vital::timestamp tstamp =
    kwiver::vital::any_cast< kwiver::vital::timestamp > ( element );
  std::stringstream msg;
  msg << "timestamp ";
  {
    cereal::JSONOutputArchive ar( msg );
    save( ar, tstamp );
  }
  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any
timestamp
::deserialize( const std::string& message )
{
  std::stringstream msg( message );
  kwiver::vital::timestamp tstamp;

  std::string tag;
  msg >> tag;
  if ( tag != "timestamp" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"timestamp\""
               << ",  received \"" << tag << "\". Message dropped." );
  }
  else
  {
    cereal::JSONInputArchive ar( msg );
    load( ar, tstamp );
  }

  return kwiver::vital::any( tstamp );
}

} } } }
