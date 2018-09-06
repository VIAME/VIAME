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
image_memory::
image_memory()
{
  m_element_names.insert( DEFAULT_ELEMENT_NAME );
}


image_memory::
~image_memory()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
image_memory::
serialize( const serialize_param_t& elements )
{
  // Get native data type from any
  kwiver::vital::image_memory obj =
    kwiver::vital::any_cast< kwiver::vital::image_memory > ( elements.at( DEFAULT_ELEMENT_NAME ) );

  std::stringstream msg;
  msg << "image_memory ";
  {
    cereal::JSONOutputArchive ar( msg );
    save( ar, obj );
  }

  return std::make_shared< std::string > ( msg.str() );
}


// ----------------------------------------------------------------------------
vital::algo::data_serializer::deserialize_result_t
image_memory::
deserialize( std::shared_ptr< std::string > message )
{
  std::stringstream msg(*message);
  kwiver::vital::image_memory img_mem;

  std::string tag;
  msg >> tag;

  if (tag != "image_memory" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"image_memory\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    cereal::JSONInputArchive ar( msg );
    load( ar, img_mem );
  }

  deserialize_result_t res;
  res[ DEFAULT_ELEMENT_NAME ] = kwiver::vital::any(img_mem);

  return res;
}

// ----------------------------------------------------------------------------
void
image_memory::
save( cereal::JSONOutputArchive& archive, kwiver::vital::image_memory& obj )
{
  std::string image_data = std::string( static_cast< char* >(obj.data()) );
  archive( cereal::make_nvp( "data", image_data ),
           cereal::make_nvp( "size", obj.size() ) );

}

// ----------------------------------------------------------------------------
void
image_memory::
load( cereal::JSONInputArchive& archive, kwiver::vital::image_memory& obj )
{
  // deserialize image_memory
  std::string data;
  std::size_t size;

  archive( CEREAL_NVP( data ),
           CEREAL_NVP( size ) );


  obj = kwiver::vital::image_memory(size);
  std::memcpy(obj.data(), data.c_str(), data.size());
}

} } } }       // end namespace kwiver
