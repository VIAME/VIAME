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

#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/types/vector.hpp>
#include <vital/internal/cereal/archives/binary.hpp>
#include <vital/internal/cereal/archives/json.hpp>

#include <zlib.h>
#include <sstream>
#include <cstddef>
#include <cstdint>

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

// ----------------------------------------------------------------------------
void
image::
save( cereal::JSONOutputArchive& archive, const kwiver::vital::image_container_sptr ctr )
{
  kwiver::vital::image vital_image = ctr->get_image();

  // Compress raw pixel data
  const uLongf size = compressBound( vital_image.size() );
  uLongf out_size(size);
  std::vector<uint8_t> image_data( size );
  Bytef* out_buf = reinterpret_cast< Bytef* >( &image_data[0] );
  Bytef const* in_buf = reinterpret_cast< Bytef * >(vital_image.memory()->data());

  int z_rc = compress( out_buf, &out_size, // outputs
                       in_buf, vital_image.size() ); // inputs
  if (Z_OK != z_rc )
  {
    switch (z_rc)
    {
    case Z_MEM_ERROR:
      LOG_ERROR( kwiver::vital::get_logger( "data_serializer" ),
                 "Error compressing image data. Not enough memory." );
      break;

    case Z_BUF_ERROR:
      LOG_ERROR( kwiver::vital::get_logger( "data_serializer" ),
                 "Error compressing image data. Not enough room in output buffer." );
      break;

    default:
      LOG_ERROR( kwiver::vital::get_logger( "data_serializer" ),
                 "Error compressing image data." );
      break;
    } // end switch
    return;
  }

  // Copy compressed image to vector
  uint8_t* cp = static_cast< uint8_t* >(out_buf );
  uint8_t* cpe = cp +out_size;
  image_data.assign( cp, cpe );

  archive( cereal::make_nvp( "width",  vital_image.width() ),
           cereal::make_nvp( "height", vital_image.height() ),
           cereal::make_nvp( "depth",  vital_image.depth() ),

           cereal::make_nvp( "w_step", vital_image.w_step() ),
           cereal::make_nvp( "h_step", vital_image.h_step() ),
           cereal::make_nvp( "d_step", vital_image.d_step() ),

           cereal::make_nvp( "img_size", vital_image.size() ), // uncompressed size
           cereal::make_nvp( "img_data", image_data ) // compressed image
    );
}

// ----------------------------------------------------------------------------
void
image::
load( cereal::JSONInputArchive& archive, kwiver::vital::image_container_sptr& ctr )
{
  // deserialize image
  std::size_t width, height, depth, img_size;
  std::ptrdiff_t w_step, h_step, d_step;
  std::vector<uint8_t> img_data;

  archive( CEREAL_NVP( width ),
           CEREAL_NVP( height ),
           CEREAL_NVP( depth ),

           CEREAL_NVP( w_step ),
           CEREAL_NVP( h_step ),
           CEREAL_NVP( d_step ),

           CEREAL_NVP( img_size ), // uncompressed size
           CEREAL_NVP( img_data )  // compressed image
    );


  auto img_mem = std::make_shared< kwiver::vital::image_memory >( img_size );

  // decompress the data
  Bytef* out_buf = reinterpret_cast< Bytef* >(img_mem->data());
  uLongf out_size( img_size );
  Bytef const* in_buf = reinterpret_cast< const Bytef *> (&img_data[0] );
  uLongf in_size = img_data.size(); // byte size

  int z_rc = uncompress( out_buf, &out_size, // outputs
                         in_buf, in_size ); // inputs
  if (Z_OK != z_rc )
  {
    switch (z_rc)
    {
    case Z_MEM_ERROR:
      LOG_ERROR( kwiver::vital::get_logger( "data_serializer" ),
                 "Error decompressing image data. Not enough memory." );
      break;

    case Z_BUF_ERROR:
      LOG_ERROR( kwiver::vital::get_logger( "data_serializer" ),
                 "Error decompressing image data. Not enough room in output buffer." );
      break;

    default:
      LOG_ERROR( kwiver::vital::get_logger( "data_serializer" ),
                 "Error decompressing image data." );
      break;
    } // end switch
    return;
  }

  if ( static_cast< uLongf >(img_size) != out_size )
  {
    LOG_ERROR( kwiver::vital::get_logger( "data_serializer" ),
               "Uncompressed data not expected size. Possible data corruption." );
    return;
  }

  auto vital_image = kwiver::vital::image( img_mem, img_mem->data(),
                                           width, height, depth,
                                           w_step, h_step, d_step);

  // return newly constructed image container
  ctr = std::make_shared< kwiver::vital::simple_image_container >( vital_image );
}

} } } }       // end namespace kwiver
