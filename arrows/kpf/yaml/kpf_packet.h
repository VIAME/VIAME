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

#include <arrows/kpf/yaml/kpf_canonical_types.h>

namespace kwiver {
namespace vital {
namespace kpf {

/**
 * @brief KPF packet styles.
 *
 */

enum class KPF_YAML_EXPORT packet_style
{
  INVALID,  // invalid, uninitialized
  META,     // an uninterpreted string (consumes all following tokens)
  ID,       // a numeric identifier (detection, track, event ID)
  TS,       // timestamp
  TSR,      // timestamp range
  LOC,      // location (2d / 3d)
  GEOM,     // bounding box
  POLY,     // polygon
  CONF,     // a confidence value
  ACT,      // an activity
  EVAL,     // an evaluation result
  ATTR,     // an attribute
  TAG,      // a tag
  KV        // a generic key/value pair
};

/**
 * @brief The KPF packet header.
 *
 * This is what combines the semantic type (the style) with the user-specified context
 * (the domain).
 *
 */

struct KPF_YAML_EXPORT packet_header_t
{
  enum { NO_DOMAIN = -1 };

  packet_style style;
  int domain;
  packet_header_t(): style( packet_style::INVALID ), domain( NO_DOMAIN ) {}
  packet_header_t( packet_style s, int d ): style(s), domain(d) {}
  explicit packet_header_t( packet_style s ): style(s), domain( NO_DOMAIN ) {}
};

/**
 * @brief Class for sorting packets based on their header.
 *
 * First check if styles are the same, then if domains are the same.
 *
 */

class KPF_YAML_EXPORT packet_header_cmp
{
public:
  bool operator()( const packet_header_t& lhs, const packet_header_t& rhs ) const;
};

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
    canonical::poly_t poly;
    canonical::meta_t meta;
    canonical::activity_t activity;
  };
  packet_t(): header( packet_header_t() ) {}
  packet_t( const packet_header_t& h ): header(h) {}
  ~packet_t();
  packet_t( const packet_t& other );
  packet_t& operator=( const packet_t& other );
};

KPF_YAML_EXPORT std::ostream& operator<<( std::ostream& os, const packet_header_t& p );
KPF_YAML_EXPORT std::ostream& operator<<( std::ostream& os, const packet_t& p );

/**
 * @brief Utility functions to convert styles to strings and vice versa.
 *
 */

KPF_YAML_EXPORT packet_style str2style( const std::string& s );
KPF_YAML_EXPORT std::string style2str( packet_style );

} // ...kpf
} // ...vital
} // ...kwiver

#endif
