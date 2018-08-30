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

#include "protobuf_util.h"
namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

std::shared_ptr< std::string >
grab_proto_from_stream( std::istringstream& msg )
{
  // get payload length
  size_t proto_size;
  msg >> proto_size;
  msg.get();  // Eat the delimiter
  
  int stream_index = static_cast<int> (msg.tellg()); 
  // Make sure the whole payload is in the stream buffer
  if ( (stream_index + proto_size) > msg.str().size() )
  {
    // THROW something TBD 
  }

  // copy payload from stream to return string.
  auto ret_data = new std::string ( msg.str().substr( stream_index, 
                                                      proto_size ) );
  std::shared_ptr< std::string >ret_string( ret_data );
  // Remove payload from stream
  msg.seekg( stream_index + proto_size);

  return ret_string;
}

} } } } // end namespace
