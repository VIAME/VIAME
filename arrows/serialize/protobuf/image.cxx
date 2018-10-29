/*ckwg +30
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

#include <vital/exceptions.h>
#include <vital/types/image_container.h>
#include <vital/util/hex_dump.h>

#include <zlib.h>
#include <cstddef>
#include <cstring>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// --------------------------------------------------------------------------
image::image()
{
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}


image::~image()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
image::
serialize( const vital::any& element )
{
  kwiver::vital::image_container_sptr img_sptr =
    kwiver::vital::any_cast< kwiver::vital::image_container_sptr > ( element );

  std::ostringstream msg;
  msg << "image ";   // add type tag
  kwiver::protobuf::image proto_img;
  convert_protobuf( img_sptr, proto_img );


  if ( ! proto_img.SerializeToOstream( &msg ) )
  {
    VITAL_THROW( kwiver::vital::serialization_exception,
                 "Error serializing detected_object_set from protobuf" );
  }

  return std::make_shared< std::string > ( msg.str() );
}


// ----------------------------------------------------------------------------
vital::any
image::
deserialize( const std::string& message )
{
  kwiver::vital::image_container_sptr img_container_sptr;
  std::istringstream msg( message );

  std::string tag;
  msg >> tag;
  msg.get();    // Eat delimiter

  if ( tag != "image" )
  {
    LOG_ERROR(
      logger(), "Invalid data type tag received. Expected \"image\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    // define our protobuf
    kwiver::protobuf::image proto_img;
    if ( ! proto_img.ParseFromIstream(&msg) )
    {
      VITAL_THROW(kwiver::vital::serialization_exception,
                  "Error deserializing image_container from protobuf");
    }

    convert_protobuf(proto_img, img_container_sptr);
  }

  return kwiver::vital::any( img_container_sptr );
}


// ----------------------------------------------------------------------------
void
image::
convert_protobuf( const kwiver::protobuf::image&      proto_img,
                  kwiver::vital::image_container_sptr& img )
{
  const size_t img_size( proto_img.size() );
  auto mem_sptr = std::make_shared<vital::image_memory>( img_size );

  // decompress the data
  uLongf out_size( img_size );
  Bytef* out_buf = reinterpret_cast< Bytef* >( mem_sptr->data() );
  Bytef const *in_buf = reinterpret_cast< Bytef const* >( proto_img.data().data() );
  uLongf in_size = proto_img.data().size(); // compressed size

  int z_rc = uncompress(out_buf, &out_size, // outputs
                        in_buf, in_size);   // inputs
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

  if (static_cast<uLongf>( img_size ) != out_size)
  {
    LOG_ERROR( kwiver::vital::get_logger( "data_serializer" ),
               "Uncompressed data not expected size. Possible data corruption.");
    return;
  }

  // create pixel trait
  const kwiver::vital::image_pixel_traits pix_trait(
    static_cast<kwiver::vital::image_pixel_traits::pixel_type>(proto_img.trait_type() ),
    proto_img.trait_num_bytes() );

  // create the image
  auto vital_image = kwiver::vital::image(
    mem_sptr, mem_sptr->data(),
    static_cast< std::size_t > ( proto_img.width() ),
    static_cast< std::size_t > ( proto_img.height() ),
    static_cast< std::size_t > ( proto_img.depth() ),
    static_cast< std::ptrdiff_t > ( proto_img.w_step() ),
    static_cast< std::ptrdiff_t > ( proto_img.h_step() ),
    static_cast< std::ptrdiff_t > ( proto_img.d_step() ),
    pix_trait
    );

  // return newly constructed image container
  img = std::make_shared< kwiver::vital::simple_image_container >( vital_image );
}


// ----------------------------------------------------------------------------
void
image::
convert_protobuf( const kwiver::vital::image_container_sptr img,
                  kwiver::protobuf::image&                  proto_img )
{
  auto vital_image = img->get_image();

  // Compress raw pixel data
  const uLongf size = compressBound( vital_image.size() );
  uLongf out_size(size);
  std::vector<uint8_t> image_data( size );
  Bytef out_buf[size];
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

  proto_img.set_width( static_cast< int64_t > ( img->width() ) );
  proto_img.set_height( static_cast< int64_t > ( img->height() ) );
  proto_img.set_depth( static_cast< int64_t > ( img->depth() ) );

  proto_img.set_w_step( static_cast< int64_t > ( vital_image.w_step() ) );
  proto_img.set_h_step( static_cast< int64_t > ( vital_image.h_step() ) );
  proto_img.set_d_step( static_cast< int64_t > ( vital_image.d_step() ) );

  // Get pixel trait
  auto pixel_trait = vital_image.pixel_traits();
  proto_img.set_trait_type( pixel_trait.type );
  proto_img.set_trait_num_bytes( pixel_trait.num_bytes );

  proto_img.set_size( img->size() ); // uncompressed size
  proto_img.set_data( out_buf, size );
}


}
}
}
}
