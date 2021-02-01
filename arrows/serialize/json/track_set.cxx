// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "track_set.h"

#include "load_save.h"
#include "load_save_track_state.h"
#include "load_save_track_set.h"

#include <vital/types/track_set.h>
#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>

#include <sstream>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

// ----------------------------------------------------------------------------
track_set::
track_set()
{ }

track_set::
~track_set()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
track_set::
serialize( const vital::any& element )
{
  kwiver::vital::track_set_sptr trk_set_sptr =
    kwiver::vital::any_cast< kwiver::vital::track_set_sptr > ( element );

  std::stringstream msg;
  msg << "track_set "; // add type tag
  {
    cereal::JSONOutputArchive ar( msg );
    save( ar, trk_set_sptr );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any track_set::
deserialize( const std::string& message )
{
  std::stringstream msg(message);
  auto trk_set_sptr = std::make_shared<kwiver::vital::track_set >();
  std::string tag;
  msg >> tag;

  if (tag != "track_set" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"track_set\", received \""
               << tag << "\". Message dropped, returning default object." );
  }
  else
  {
    cereal::JSONInputArchive ar( msg );
    load( ar, trk_set_sptr );
  }

  return kwiver::vital::any( trk_set_sptr );
}

} } } }       // end namespace kwiver
