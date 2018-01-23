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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
 * @file
 * @brief The KPF packet type.
 *
 * The packet has two parts: a header (style / domain), and the payload.
 * The payload is an instance of one of the KPF canonical types.
 *
 */

#ifndef KWIVER_VITAL_KPF_PACKET_H_
#define KWIVER_VITAL_KPF_PACKET_H_

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <arrows/kpf/yaml/kpf_yaml_export.h>
#include <arrows/kpf/yaml/kpf_packet_header.h>
#include <arrows/kpf/yaml/kpf_canonical_types.h>

namespace kwiver {
namespace vital {
namespace kpf {

/**
 * @brief The KPF packet.
 *
 *
 */

struct KPF_YAML_EXPORT packet_t
{
  packet_header_t header;
  union
  {
    canonical::id_t id;
    canonical::timestamp_t timestamp;
    canonical::timestamp_range_t timestamp_range;
    canonical::bbox_t bbox;
    canonical::kv_t kv;
    canonical::conf_t conf;
    canonical::cset_t* cset;
    canonical::poly_t poly;
    canonical::meta_t meta;
    canonical::eval_t eval;
    canonical::activity_t activity;
  };
  packet_t(): header( packet_header_t() ) {}
  packet_t( const packet_header_t& h );
  ~packet_t();
  packet_t( const packet_t& other );
  packet_t& operator=( const packet_t& other );

  packet_t( packet_t&& other);
  packet_t& operator=( packet_t&& other );
};

KPF_YAML_EXPORT std::ostream& operator<<( std::ostream& os, const packet_t& p );


} // ...kpf
} // ...vital
} // ...kwiver

#endif
