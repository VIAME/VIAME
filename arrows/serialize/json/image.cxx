/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "image.h"
#include "image_memory.h"

#include <vital/types/image.h>
#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>

#include <sstream>
#include <cstddef>

namespace kasj = kwiver::arrows::serialize::json;

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

// ----------------------------------------------------------------------------
image::
image()
{
  m_element_names.insert( DEFAULT_ELEMENT_NAME );
}


image::
~image()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
image::
serialize( const serialize_param_t& elements )
{
  // Get native data type from any
  kwiver::vital::image obj =
    kwiver::vital::any_cast< kwiver::vital::image > ( elements.at( DEFAULT_ELEMENT_NAME ) );

  std::stringstream msg;
  msg << "image ";
  {
    cereal::JSONOutputArchive ar( msg );
    save( ar, obj );
  }

  return std::make_shared< std::string > ( msg.str() );
}


// ----------------------------------------------------------------------------
vital::algo::data_serializer::deserialize_result_t
image::
deserialize( std::shared_ptr< std::string > message )
{
  std::stringstream msg(*message);
  kwiver::vital::image img;

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
    load( ar, img );
  }

  deserialize_result_t res;
  res[ DEFAULT_ELEMENT_NAME ] = kwiver::vital::any(img);

  return res;
}

// ----------------------------------------------------------------------------
void
image::
save( cereal::JSONOutputArchive& archive, kwiver::vital::image& obj )
{
  kasj::image_memory::save( archive, *obj.memory() );
  archive( cereal::make_nvp( "width", obj.width() ),
           cereal::make_nvp( "height", obj.height() ), 
           cereal::make_nvp( "depth", obj.depth() ) );

}

// ----------------------------------------------------------------------------
void
image::
load( cereal::JSONInputArchive& archive, kwiver::vital::image& obj )
{
  // deserialize image
  std::size_t width, height, depth;

  archive( CEREAL_NVP( width ),
           CEREAL_NVP( height ),
           CEREAL_NVP( depth ) );

  obj = kwiver::vital::image(width, height, depth);
  kasj::image_memory::load(archive, *obj.memory());
}

} } } }       // end namespace kwiver
