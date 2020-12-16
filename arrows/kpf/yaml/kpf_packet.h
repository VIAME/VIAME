// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
