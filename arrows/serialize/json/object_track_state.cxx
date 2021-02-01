// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "object_track_state.h"

#include <arrows/serialize/json/load_save.h>
#include <arrows/serialize/json/load_save_track_state.h>

#include <vital/types/object_track_set.h>
#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>

#include <sstream>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

// ----------------------------------------------------------------------------
object_track_state::
object_track_state()
{ }

object_track_state::
~object_track_state()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
object_track_state::
serialize( const vital::any& element )
{
  kwiver::vital::object_track_state obj_trk_state =
    kwiver::vital::any_cast< kwiver::vital::object_track_state > ( element );

  std::stringstream msg;
  msg << "object_track_state "; // add type tag
  {
    cereal::JSONOutputArchive ar( msg );
    save( ar, obj_trk_state );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any object_track_state::
deserialize( const std::string& message )
{
  std::stringstream msg(message);
  kwiver::vital::object_track_state obj_trk_state;
  std::string tag;
  msg >> tag;

  if (tag != "object_track_state" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"track_state\", received \""
               << tag << "\". Message dropped, returning default object." );
  }
  else
  {
    cereal::JSONInputArchive ar( msg );
    load( ar, obj_trk_state );
  }

  return kwiver::vital::any( obj_trk_state );
}

} } } }       // end namespace kwiver
