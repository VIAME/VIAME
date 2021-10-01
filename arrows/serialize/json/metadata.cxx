// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "metadata.h"
#include "load_save.h"

#include <vital/types/metadata.h>
#include <vital/types/metadata_map.h>
#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>

#include <sstream>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

// ----------------------------------------------------------------------------
metadata
::metadata()
{
}

// ----------------------------------------------------------------------------
metadata
::~metadata()
{
}

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
metadata
::serialize_meta( kwiver::vital::metadata_vector const& meta )
{
  std::stringstream msg;
  msg << "metadata "; // add type tag
  {
    cereal::JSONOutputArchive ar( msg );
    save( ar, meta );
  }

  return std::make_shared< std::string >( msg.str() );
}

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
metadata
::serialize_map( vital::metadata_map::map_metadata_t const& frame_map )
{
  std::stringstream msg;
  {
    cereal::JSONOutputArchive ar( msg );
    save( ar, frame_map );
  }

  return std::make_shared< std::string >( msg.str() );
}

// ----------------------------------------------------------------------------
vital::metadata_map::map_metadata_t
metadata::
deserialize_map( const std::string& message )
{
  std::stringstream msg( message );
  kwiver::vital::metadata_map::map_metadata_t metadata;

  cereal::JSONInputArchive ar( msg );
  load( ar, metadata );

  return metadata;
}

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
metadata::
serialize( const vital::any& element )
{
  const kwiver::vital::metadata_vector meta =
    kwiver::vital::any_cast< kwiver::vital::metadata_vector > ( element );

  std::stringstream msg;
  msg << "metadata "; // add type tag
  {
    cereal::JSONOutputArchive ar( msg );
    save( ar, meta );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any metadata::
deserialize( const std::string& message )
{
  std::stringstream msg(message);
  kwiver::vital::metadata_vector meta;
  std::string tag;
  msg >> tag;

  if (tag != "metadata" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"metadata\", received \""
               << tag << "\". Message dropped, returning default object." );
  }
  else
  {
    cereal::JSONInputArchive ar( msg );
    load( ar, meta );
  }

  return kwiver::vital::any(meta);
}

} } } }       // end namespace kwiver
