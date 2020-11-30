// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
