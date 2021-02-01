// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "bounding_box.h"
#include "load_save.h"

#include <vital/types/bounding_box.h>
#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>

#include <sstream>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

// ----------------------------------------------------------------------------
bounding_box::
bounding_box()
{ }

bounding_box::
~bounding_box()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
bounding_box::
serialize( const vital::any& element )
{
  kwiver::vital::bounding_box_d bbox =
    kwiver::vital::any_cast< kwiver::vital::bounding_box_d > ( element );

  std::stringstream msg;
  msg << "bounding_box "; // add type tag
  {
    cereal::JSONOutputArchive ar( msg );
    save( ar, bbox );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any bounding_box::
deserialize( const std::string& message )
{
  std::stringstream msg(message);
  kwiver::vital::bounding_box_d bbox{ 0, 0, 0, 0 };
  std::string tag;
  msg >> tag;

  if (tag != "bounding_box" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"bounding_box\", received \""
               << tag << "\". Message dropped, returning default object." );
  }
  else
  {
    cereal::JSONInputArchive ar( msg );
    load( ar, bbox );
  }

  return kwiver::vital::any(bbox);
}

} } } }       // end namespace kwiver
