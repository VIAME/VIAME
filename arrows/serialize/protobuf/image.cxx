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
#include "image_memory.h"
#include <vital/exceptions.h>
#include <cstddef>
#include <cstring>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {
  // --------------------------------------------------------------------------
  image::image()
  {
    m_element_names.insert( DEFAULT_ELEMENT_NAME );

    // Verify that the version of the library that we linked against is
    // compatible with the version of the headers we compiled against.
    GOOGLE_PROTOBUF_VERIFY_VERSION;
  }

  image::~image()
  { }

  // ----------------------------------------------------------------------------
  std::shared_ptr< std::string >
  image::
  serialize( const data_serializer::serialize_param_t& elements )
  {
    kwiver::vital::image img =
      kwiver::vital::any_cast< kwiver::vital::image > ( elements.at( DEFAULT_ELEMENT_NAME ) );

    std::ostringstream msg;
    msg << "image "; // add type tag

    kwiver::protobuf::image proto_img;
    convert_protobuf( img, proto_img );

    if ( ! proto_img.SerializeToOstream( &msg ) )
    {
      VITAL_THROW( kwiver::vital::serialization_exception,
                   "Error serializing detected_object_set from protobuf" );
    }

    return std::make_shared< std::string > ( msg.str() );
  }

  // ----------------------------------------------------------------------------
  vital::algo::data_serializer::deserialize_result_t
  image::
  deserialize( std::shared_ptr< std::string > message )
  {
    kwiver::vital::image img; 
    std::istringstream msg( *message );

    std::string tag;
    msg >> tag;
    msg.get();  // Eat delimiter

    if (tag != "image" )
    {
      LOG_ERROR( logger(), "Invalid data type tag received. Expected \"image\", received \""
                 << tag << "\". Message dropped." );
    }
    else
    {
      // define our protobuf
      kwiver::protobuf::image proto_img;
      if ( ! proto_img.ParseFromIstream( &msg ) )
      {
        VITAL_THROW( kwiver::vital::serialization_exception,
                     "Error deserializing detected_object_set from protobuf" );
      }

      convert_protobuf( proto_img, img );
    }

    deserialize_result_t res;
    res[ DEFAULT_ELEMENT_NAME ] = kwiver::vital::any(img);

    return res;
  }

  // ----------------------------------------------------------------------------
  void image::
  convert_protobuf( const kwiver::protobuf::image&  proto_img,
                    kwiver::vital::image& img )
  {
    img = kwiver::vital::image(static_cast< size_t >( proto_img.width() ),
                               static_cast< size_t >( proto_img.height() ),
                               static_cast< size_t >( proto_img.depth() ) );
    kwiver::arrows::serialize::protobuf::image_memory::convert_protobuf( 
        proto_img.data(), *img.memory() );
  }

  // ----------------------------------------------------------------------------
  void image::
  convert_protobuf( const kwiver::vital::image& img,
                    kwiver::protobuf::image&  proto_img )
  {
    proto_img.set_width( static_cast< int64_t >( img.width()) );
    proto_img.set_height( static_cast< int64_t >( img.height() ) );
    proto_img.set_depth( static_cast< int64_t >( img.depth() ) );
    
    kwiver::protobuf::image_memory *proto_img_mem = new kwiver::protobuf::image_memory();
    kwiver::arrows::serialize::protobuf::image_memory::convert_protobuf( 
                                       *img.memory(), *proto_img_mem);
    proto_img.set_allocated_data(proto_img_mem);
  }

} } } }
