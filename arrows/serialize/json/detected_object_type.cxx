// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "detected_object_type.h"

#include "load_save.h"

#include <vital/types/detected_object_type.h>
#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>

#include <sstream>
namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

// ----------------------------------------------------------------------------
detected_object_type::
detected_object_type()
{ }

detected_object_type::
~detected_object_type()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
detected_object_type::
serialize( const vital::any& element )
{
  kwiver::vital::detected_object_type dot =
    kwiver::vital::any_cast< kwiver::vital::detected_object_type > ( element );

  std::stringstream msg;
  msg << "detected_object_type ";
  {
    cereal::JSONOutputArchive ar( msg );
    save( ar, dot );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any detected_object_type::
deserialize( const std::string& message )
{
  std::stringstream msg(message);
  kwiver::vital::detected_object_type dot;
  std::string tag;
  msg >> tag;

  if (tag != "detected_object_type" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"detected_object_type\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    cereal::JSONInputArchive ar( msg );
    load( ar, dot );
  }

  return kwiver::vital::any(dot);
}

} } } }       // end namespace kwiver
