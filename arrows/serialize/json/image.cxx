// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "image.h"
#include "load_save.h"

#include <vital/types/image_container.h>

#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>

#include <sstream>

namespace kasj = kwiver::arrows::serialize::json;

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

// ----------------------------------------------------------------------------
image::
image()
{ }

image::
~image()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
image::
serialize( const vital::any& element )
{
  // Get native data type from any
  kwiver::vital::image_container_sptr obj =
    kwiver::vital::any_cast< kwiver::vital::image_container_sptr > ( element );

  std::stringstream msg;
  msg << "image ";
  {
    cereal::JSONOutputArchive ar( msg );
    save( ar, obj );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any
image::
deserialize( const std::string& message )
{
  std::stringstream msg(message);
  kwiver::vital::image_container_sptr img_ctr_sptr;

  std::string tag;
  msg >> tag;

  if (tag != "image" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"image\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    cereal::JSONInputArchive ar( msg );
    load( ar, img_ctr_sptr );
  }

  return kwiver::vital::any( img_ctr_sptr );
}

} } } }       // end namespace kwiver
