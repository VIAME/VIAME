// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "string.h"

#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>

#include <sstream>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

string::string()
{ }

string::~string()
{ }

std::shared_ptr< std::string > string::
serialize( const vital::any& element )
{
  const std::string data = kwiver::vital::any_cast< std::string >( element );
  std::stringstream msg;
  msg << "string ";
  {
    cereal::JSONOutputArchive ar( msg );
    save( ar, data );
  }  

  return std::make_shared< std::string > ( msg.str() );
}

vital::any string::deserialize( const std::string& message )
{
  std::string content, tag;
  std::istringstream msg( message );
  msg >> tag;

  if (tag != "string")
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"string\", received \""
            << tag << "\". Message dropped.");
  }
  else
  {
    cereal::JSONInputArchive ar(msg);
    load( ar, content );
  }
  return kwiver::vital::any(content);
}

void string::save( cereal::JSONOutputArchive& archive, const std::string& str )
{
  archive( CEREAL_NVP( str ) );
}

void string::load( cereal::JSONInputArchive& archive,  std::string& str)
{
  archive( CEREAL_NVP( str ) );
}

} } } }     // end namespace kwiver
