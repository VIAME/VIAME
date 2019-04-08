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
 * @brief The KPF packet header type.
 *
 */

#ifndef KWIVER_VITAL_KPF_PACKET_HEADER_H_
#define KWIVER_VITAL_KPF_PACKET_HEADER_H_

#include <arrows/kpf/yaml/kpf_yaml_export.h>

#include <string>

namespace kwiver {
namespace vital {
namespace kpf {

/**
 * @brief KPF packet styles.
 *
 */

enum class packet_style
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
  CSET,     // a set of label:confidence KV packes
  ACT,      // an activity
  EVAL,     // an evaluation result
  ATTR,     // an attribute
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
  enum { ANY_DOMAIN = -2, NO_DOMAIN = -1 };

  packet_style style;
  int domain;
  packet_header_t(): style( packet_style::INVALID ), domain( NO_DOMAIN ) {}
  packet_header_t( packet_style s, int d ): style(s), domain(d) {}
  explicit packet_header_t( packet_style s ): style(s), domain( NO_DOMAIN ) {}
};

KPF_YAML_EXPORT bool operator==( const packet_header_t& lhs, const packet_header_t& rhs );

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
 * @brief Utility functions to convert styles to strings and vice versa.
 *
 */

KPF_YAML_EXPORT packet_style str2style( const std::string& s );
KPF_YAML_EXPORT std::string style2str( packet_style );

/**
 * @brief output operater for a packet header
 *
 */

KPF_YAML_EXPORT std::ostream& operator<<( std::ostream& os, const packet_header_t& p );

} // ...kpf
} // ...vital
} // ...kwiver

#endif
