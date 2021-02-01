// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "detected_object_set.h"

#include "detected_object.h"
#include "load_save.h"

#include <vital/types/detected_object_set.h>

#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>

#include <sstream>

namespace kasj = kwiver::arrows::serialize::json;

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

// ----------------------------------------------------------------------------
detected_object_set::
detected_object_set()
{ }

detected_object_set::
~detected_object_set()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
detected_object_set::
serialize( const vital::any& element )
{
  kwiver::vital::detected_object_set_sptr obj =
    kwiver::vital::any_cast< kwiver::vital::detected_object_set_sptr > ( element );

  std::stringstream msg;
  msg << "detected_object_set ";
  {
    cereal::JSONOutputArchive ar( msg );
    save( ar, *obj );
  }
  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any detected_object_set::
deserialize( const std::string& message )
{
  std::stringstream msg(message);
  kwiver::vital::detected_object_set* obj =  new kwiver::vital::detected_object_set();

  std::string tag;
  msg >> tag;

  if (tag != "detected_object_set" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"detected_object_set\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    cereal::JSONInputArchive ar( msg );
    load( ar, *obj );
  }

  return kwiver::vital::any( kwiver::vital::detected_object_set_sptr( obj ) );
}

} } } }       // end namespace kwiver
