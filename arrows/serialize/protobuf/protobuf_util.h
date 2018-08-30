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

#ifndef ARROWS_SERIALIZE_PROTOBUF_UTIL_
#define ARROWS_SERIALIZE_PROTOBUF_UTIL_

#include <arrows/serialize/protobuf/kwiver_serialize_protobuf_export.h>

#include <string>
#include <sstream>
#include <memory>
namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

//
// These functions pack and unpack serialized protobufs into/from a
// byte stream.
//

/**
 * @brief Serialize protobuf to stream.
 *
 * This function adds the serialized payload to the stream. The
 * payload length, in bytes, is written to the stream followed by the
 * actual payload data.
 *
 * @param msg Stream to serialize to
 * @param proto Protobuf to serialize
 */
template<class T>
void
add_proto_to_stream( std::ostringstream& msg,
                     const T& proto )
{
  // get size of serialized protobuf
  const size_t proto_size( proto.ByteSize() );
  // Add payload size to stream
  msg << proto_size << " ";
  if ( ! proto.SerializeToOstream( &msg ) )
  {
    //+ TBD log error / throw
  }

}


// ----------------------------------------------------------------------------
/**
 * @brief Return byte payload from stream.
 *
 * This function returns the first payload from the stream. Each
 * payload is prefixed with the actual data length, in bytes. The
 * stream is updated as the payload is removed.
 *
 * @param msg Byte stream containing payload.
 *
 * @return The first payload in the stream.
 */
std::shared_ptr< std::string > KWIVER_SERIALIZE_PROTOBUF_EXPORT
grab_proto_from_stream( std::istringstream& msg );

} } } } // end namespace

#endif  // ARROWS_SERIALIZE_PROTOBUF_UTIL_
