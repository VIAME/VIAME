// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "track.h"

#include "load_save.h"
#include "load_save_track_state.h"
#include "track_item.h"

#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>
#include <vital/internal/cereal/types/utility.hpp>

#include <sstream>
#include <iostream>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {
// ----------------------------------------------------------------------------
track::
track()
{ }

track::
~track()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
track::
serialize( const vital::any& element )
{
  kwiver::vital::track_sptr trk_sptr =
    kwiver::vital::any_cast< kwiver::vital::track_sptr > ( element );
  kwiver::arrows::serialize::json::track_item trk_item(trk_sptr);
  std::stringstream msg;
  msg << "track "; // add type tag
  {
    cereal::JSONOutputArchive ar( msg );
    ar( trk_item );
  }
  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any track::
deserialize( const std::string& message )
{
  std::stringstream msg(message);
  kwiver::arrows::serialize::json::track_item trk_item = track_item();
  std::string tag;
  msg >> tag;

  if (tag != "track" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"track\", received \""
               << tag << "\". Message dropped, returning default object." );
  }
  else
  {
    cereal::JSONInputArchive ar( msg );
    ar( trk_item );
  }
  return kwiver::vital::any( trk_item.get_track() );
}

} } } }       // end namespace kwiver
