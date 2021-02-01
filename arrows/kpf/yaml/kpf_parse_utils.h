// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file
 * @brief Various utility functions for parsing KPF.
 *
 */

#ifndef KWIVER_VITAL_KPR_PARSE_UTILS_H_
#define KWIVER_VITAL_KPR_PARSE_UTILS_H_

#include <arrows/kpf/yaml/kpf_packet.h>

#include <string>
#include <vector>
#include <tuple>
#include <map>

namespace kwiver {
namespace vital {
namespace kpf {

/**
 * @brief This maps KPF packet headers to their full packets.
 *
 * The packet buffer holds the parsed KPF packets for the current line (aka record.)
 * Packets are transferred out of the buffer to the client via the kfp_reader.
 *
 * The packet buffer is a multimap because some packets may appear multiple
 * times (i.e. key-value packets.)
 */

typedef std::multimap< packet_header_t,
                       packet_t,
                       packet_header_cmp > packet_buffer_t;

typedef std::multimap< packet_header_t,
                       packet_t,
                       packet_header_cmp >::const_iterator packet_buffer_cit;

typedef std::tuple< bool, std::string, int > header_parse_t;

/**
 * @brief Convert a string (e.g. 'id2') into a KPF packet header.
 *
 * @return true if the conversion is successful.
 */

packet_header_t KPF_YAML_EXPORT
packet_header_parser( const std::string& s );

header_parse_t KPF_YAML_EXPORT parse_header( const std::string& s, bool expect_colon );

} // ...kpf
} // ...vital
} // ...kwiver

#endif
