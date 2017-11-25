/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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

/**
 * \file
 * \brief Bounce buffer.
 *
 *
 * This is the bounce buffer between the parser's
 * packet buffer and the user. This could probably be
 * refactored away.
 *
 */

#ifndef KWIVER_VITAL_KPF_BOUNCE_BUFFER_H_
#define KWIVER_VITAL_KPF_BOUNCE_BUFFER_H_

#include <arrows/kpf/yaml/kpf_packet.h>

#include <string>
#include <utility>

namespace kwiver {
namespace vital {
namespace kpf {

class KPF_YAML_EXPORT packet_bounce_t
{
public:
  packet_bounce_t();
  explicit packet_bounce_t( const std::string& tag );
  explicit packet_bounce_t( const packet_header_t& h );
  void init( const std::string& tag );
  void init( const packet_header_t& h );
  ~packet_bounce_t() {}

  // mutate the domain
  packet_bounce_t& set_domain( int d );

  // return this reader's packet header
  packet_header_t my_header() const;

  // transfer packet into the reader
  void set_from_buffer( const packet_t& );

  // return (true, packet) and clear the is_set flag
  // return false if set_from_buffer hasn't been called yet
  std::pair< bool, packet_t > get_packet();

protected:
  bool is_set;
  packet_header_t header;
  packet_t packet;
};

} // ...kpf
} // ...vital
} // ...kwiver


#endif

