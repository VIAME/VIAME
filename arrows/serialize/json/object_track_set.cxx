// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "object_track_set.h"

#include "load_save.h"
#include "load_save_track_state.h"
#include "load_save_track_set.h"

#include <vital/types/object_track_set.h>
#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>

#include <sstream>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

// ----------------------------------------------------------------------------
object_track_set::
object_track_set()
{ }

object_track_set::
~object_track_set()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
object_track_set::
serialize( const vital::any& element )
{
  kwiver::vital::object_track_set_sptr obj_trk_set_sptr =
    kwiver::vital::any_cast< kwiver::vital::object_track_set_sptr > ( element );

  std::stringstream msg;
  msg << "object_track_set "; // add type tag
  {
    cereal::JSONOutputArchive ar( msg );
    save( ar, obj_trk_set_sptr );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any object_track_set::
deserialize( const std::string& message )
{
  std::stringstream msg(message);
  auto obj_trk_set_sptr = std::make_shared<kwiver::vital::object_track_set >();
  std::string tag;
  msg >> tag;

  if (tag != "object_track_set" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. "
                << " Expected \"object_track_set\", received \""
               << tag << "\". Message dropped, returning default object." );
  }
  else
  {
    cereal::JSONInputArchive ar( msg );
    load( ar, obj_trk_set_sptr );
  }

  return kwiver::vital::any( obj_trk_set_sptr );
}

} } } }       // end namespace kwiver
