// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file
 * @brief Utilities for reading and writing KPF in track_oracle.
 *
 * add_to_row() adds the KPF packet to the track_oracle row. Any
 * ad hoc fields are added with a specific naming convention which
 * is recognized by get_optional_fields().
 *
 * get_optional_fields() scans the track_oracle data columns (aka fields)
 * and returns a map of which fields correspond to KPF packets.
 *
 */

#ifndef INCL_KPF_UTILITIES_H
#define INCL_KPF_UTILITIES_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/kpf_utils/kpf_utils_export.h>

#include <track_oracle/core/track_oracle_api_types.h>
#include <track_oracle/utils/logging_map.h>

#include <arrows/kpf/yaml/kpf_packet.h>
#include <arrows/kpf/yaml/kpf_yaml_writer.h>

#include <map>

namespace kwiver {
namespace track_oracle {
namespace kpf_utils {

namespace KPF=::kwiver::vital::kpf;

KPF_UTILS_EXPORT
void
add_to_row( kwiver::logging_map_type& log_map,
            const oracle_entry_handle_type& row,
            const KPF::packet_t& p );

struct KPF_UTILS_EXPORT optional_field_state
{
  bool first_pass;
  std::map< field_handle_type, KPF::packet_t > optional_fields;
  kwiver::logging_map_type& log_map;

  explicit optional_field_state( kwiver::logging_map_type& lmt );
};

KPF_UTILS_EXPORT
std::vector< KPF::packet_t >
optional_fields_to_packets( optional_field_state& ofs,
                            const oracle_entry_handle_type& row );

KPF_UTILS_EXPORT
void
write_optional_packets( const std::vector< KPF::packet_t>& packets,
                        kwiver::logging_map_type& log_map,
                        KPF::record_yaml_writer& w );

KPF_UTILS_EXPORT
track_handle_list_type
read_unstructured_yaml( const std::string& fn );

} // ...kpf_utils
} // ...track_oracle
} // ...kwiver

#endif
