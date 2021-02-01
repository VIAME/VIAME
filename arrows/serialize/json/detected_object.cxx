// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "detected_object.h"

#include "bounding_box.h"

#include <vital/types/detected_object.h>
#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>

#include <sstream>

namespace kasj = kwiver::arrows::serialize::json;

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

// ----------------------------------------------------------------------------
detected_object::
detected_object()
{ }

detected_object::
~detected_object()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
detected_object::
serialize( const vital::any& element )
{
  // Get native data type from any
  kwiver::vital::detected_object_sptr obj =
    kwiver::vital::any_cast< kwiver::vital::detected_object_sptr > ( element );

  std::stringstream msg;
  msg << "detected_object ";
  {
    cereal::JSONOutputArchive ar( msg );
    save( ar, *obj );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any detected_object::
deserialize( const std::string& message )
{
  std::stringstream msg(message);
  auto obj = std::make_shared< kwiver::vital::detected_object >( kwiver::vital::bounding_box_d { 0, 0, 0, 0 } );

  std::string tag;
  msg >> tag;

  if (tag != "detected_object" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"detected_object\", received \""
               << tag << "\". Message dropped. Default object returned." );
  }
  else
  {
    cereal::JSONInputArchive ar( msg );
    load( ar, *obj );
  }

  return kwiver::vital::any(obj);
}

} } } }       // end namespace kwiver
