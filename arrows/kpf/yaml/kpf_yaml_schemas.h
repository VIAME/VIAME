// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file
 * @brief Support for predefined KPF yaml schemas.
 *
 * The KPF schema is mostly a validation aid, associating a tag ("geom",
 * "act") with a set of required packet types.
 *
 */

#ifndef KWIVER_VITAL_KPF_YAML_SCHEMAS_H_
#define KWIVER_VITAL_KPF_YAML_SCHEMAS_H_

#include <arrows/kpf/yaml/kpf_parse_utils.h>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace kwiver {
namespace vital {
namespace kpf {

enum class schema_style {
  INVALID,     // invalid
  META,        // metadata
  GEOM,        // geometry
  ACT,         // activity
  TYPES,       // object types
  REGIONS,     // regions file
  UNSPECIFIED  // no validation provided
};

struct KPF_YAML_EXPORT validation_data
{
  packet_style style;
  std::string key;
  validation_data(): style( packet_style::INVALID ), key("") {}
  explicit validation_data( YAML::const_iterator it );
  explicit validation_data( packet_style s ): style(s), key("") {}
  validation_data( packet_style s, const std::string& k ): style(s), key(k) {}

  static std::string schema_style_to_str( schema_style s );
  static schema_style str_to_schema_style( const std::string& s );
};

std::vector< validation_data > KPF_YAML_EXPORT
validate_schema( schema_style schema, const packet_buffer_t& packets );

std::vector< validation_data > KPF_YAML_EXPORT
validate_schema( schema_style schema, const std::vector< validation_data>& vpackets );

} // ...kpf
} // ...vital
} // ...kwiver

#endif
