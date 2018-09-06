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


#ifndef ARROWS_SERIALIZATION_PROTO_IMAGE_MEMORY_H
#define ARROWS_SERIALIZATION_PROTO_IMAGE_MEMORY_H

#include <arrows/serialize/protobuf/kwiver_serialize_protobuf_export.h>
#include <vital/algo/data_serializer.h>
#include <vital/types/image.h>
#include <vital/types/protobuf/image_memory.pb.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

  class KWIVER_SERIALIZE_PROTOBUF_EXPORT image_memory
  : public vital::algorithm_impl< image_memory, vital::algo::data_serializer >
  {
    public:
      static constexpr char const* name = "kwiver:image_memory";
      static constexpr char const* description = 
        "Serializes an image memory representation using protobuf notation. "
        "This implementation handles only a single data item.";
      
      image_memory();
      virtual ~image_memory();
      virtual std::shared_ptr< std::string > serialize( const serialize_param_t& elements );
      virtual deserialize_result_t deserialize( std::shared_ptr< std::string > message );

      // Convert between native and protobuf formats
      static void convert_protobuf( const kwiver::protobuf::image_memory&  proto_img_mem,
                                    kwiver::vital::image_memory& img_mem);

      static void convert_protobuf( kwiver::vital::image_memory& img_mem,
                                    kwiver::protobuf::image_memory&  proto_img_mem );
  };
} } } }
#endif // ARROWS_SERIALIZATION_PROTO_IMAGE_MEMORY_H 
