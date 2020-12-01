// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <arrows/serialize/json/activity.h>
#include <arrows/serialize/json/load_save.h>

#include <vital/types/activity.h>
#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>

#include <sstream>

namespace kasj = kwiver::arrows::serialize::json;

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

// ----------------------------------------------------------------------------
activity::
activity()
{ }

activity::
~activity()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
activity::
serialize( const kwiver::vital::any& element )
{
  kwiver::vital::activity l_activity =
    kwiver::vital::any_cast< kwiver::vital::activity > ( element );

  std::stringstream msg;
  msg << "activity ";
  {
    cereal::JSONOutputArchive ar( msg );
    save( ar, l_activity );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
kwiver::vital::any activity::
deserialize( const std::string& message )
{
  std::stringstream msg(message);
  kwiver::vital::activity l_activity;
  std::string tag;
  msg >> tag;

  if (tag != "activity" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"activity\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    cereal::JSONInputArchive ar( msg );
    load( ar, l_activity );
  }

  return kwiver::vital::any( l_activity );
}

} } } }
