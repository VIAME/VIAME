// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <arrows/serialize/json/activity_type.h>
#include <arrows/serialize/json/load_save.h>

#include <vital/types/activity_type.h>
#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>

#include <sstream>

namespace kasj = kwiver::arrows::serialize::json;

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

// ----------------------------------------------------------------------------
activity_type::
activity_type()
{ }

activity_type::
~activity_type()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
activity_type::
serialize( const kwiver::vital::any& element )
{
  kwiver::vital::activity_type at =
    kwiver::vital::any_cast< kwiver::vital::activity_type > ( element );

  std::stringstream msg;
  msg << "activity_type ";
  {
    cereal::JSONOutputArchive ar( msg );
    save( ar, at );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
kwiver::vital::any activity_type::
deserialize( const std::string& message )
{
  std::stringstream msg(message);
  kwiver::vital::activity_type at;
  std::string tag;
  msg >> tag;

  if (tag != "activity_type" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"activity_type\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    cereal::JSONInputArchive ar( msg );
    load( ar, at );
  }

  return kwiver::vital::any(at);
}

} } } }
