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
#include <vital/exceptions.h>
#include <cstddef>
#include <cstring>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {
  // --------------------------------------------------------------------------
  image_memory::image_memory()
  {
    m_element_names.insert( DEFAULT_ELEMENT_NAME );

    // Verify that the version of the library that we linked against is
    // compatible with the version of the headers we compiled against.
    GOOGLE_PROTOBUF_VERIFY_VERSION;
  }

  image_memory::~image_memory()
  { }

  // ----------------------------------------------------------------------------
  std::shared_ptr< std::string >
  image_memory::
  serialize( const data_serializer::serialize_param_t& elements )
  {
    kwiver::vital::image_memory img_mem =
      kwiver::vital::any_cast< kwiver::vital::image_memory > ( elements.at( DEFAULT_ELEMENT_NAME ) );

    std::ostringstream msg;
    msg << "image_memory "; // add type tag

    kwiver::protobuf::image_memory proto_img_mem;
    convert_protobuf( img_mem, proto_img_mem );

    if ( ! proto_img_mem.SerializeToOstream( &msg ) )
    {
      VITAL_THROW( kwiver::vital::serialization_exception,
                   "Error serializing detected_object_set from protobuf" );
    }

    return std::make_shared< std::string > ( msg.str() );
  }

  // ----------------------------------------------------------------------------
  vital::algo::data_serializer::deserialize_result_t
  image_memory::
  deserialize( std::shared_ptr< std::string > message )
  {
    kwiver::vital::image_memory img_mem; 
    std::istringstream msg( *message );

    std::string tag;
    msg >> tag;
    msg.get();  // Eat delimiter

    if (tag != "image_memory" )
    {
      LOG_ERROR( logger(), "Invalid data type tag received. Expected \"image_memory\", received \""
                 << tag << "\". Message dropped." );
    }
    else
    {
      // define our protobuf
      kwiver::protobuf::image_memory proto_img_mem;
      if ( ! proto_img_mem.ParseFromIstream( &msg ) )
      {
        VITAL_THROW( kwiver::vital::serialization_exception,
                     "Error deserializing detected_object_set from protobuf" );
      }

      convert_protobuf( proto_img_mem, img_mem );
    }

    deserialize_result_t res;
    res[ DEFAULT_ELEMENT_NAME ] = kwiver::vital::any(img_mem);

    return res;
  }

  // ----------------------------------------------------------------------------
  void image_memory::
  convert_protobuf( const kwiver::protobuf::image_memory&  proto_img_mem,
                    kwiver::vital::image_memory& img_mem )
  {
    std::size_t img_size = static_cast< std::size_t >(proto_img_mem.size());
    img_mem = kwiver::vital::image_memory(img_size);
    std::memcpy(img_mem.data(), proto_img_mem.data().c_str(), 
                proto_img_mem.data().size());
  }

  // ----------------------------------------------------------------------------
  void image_memory::
  convert_protobuf( kwiver::vital::image_memory& img_mem,
                    kwiver::protobuf::image_memory&  proto_img_mem )
  {
    proto_img_mem.set_data( static_cast< char* >( img_mem.data() ), img_mem.size() );
    proto_img_mem.set_size( static_cast< int64_t >( img_mem.size() ) );
  }

} } } }
