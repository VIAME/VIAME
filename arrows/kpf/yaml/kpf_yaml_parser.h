// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file
 * @brief KPF YAML parser class.
 *
 * Header for the KPF YAML parser; holds the YAML root document and provides
 * the interface for reading each KPF line (which shows up as a YAML map)
 * into the KPF generic parser's packet buffer.
 */

#ifndef KWIVER_VITAL_KPF_YAML_PARSER_H_
#define KWIVER_VITAL_KPF_YAML_PARSER_H_

#include <arrows/kpf/yaml/kpf_parse_utils.h>
#include <arrows/kpf/yaml/kpf_parser_base.h>

#include <yaml-cpp/yaml.h>

namespace kwiver {
namespace vital {
namespace kpf {

class KPF_YAML_EXPORT kpf_yaml_parser_t: public kpf_parser_base_t
{
public:
  explicit kpf_yaml_parser_t( std::istream& is );
  ~kpf_yaml_parser_t() {}

  virtual bool get_status() const;
  virtual bool parse_next_record( packet_buffer_t& pb );
  virtual schema_style get_current_record_schema() const;
  virtual bool eof() const;

private:
  YAML::Node root;
  YAML::const_iterator current_record;
  schema_style current_record_schema;
};

} // ...kpf
} // ...vital
} // ...kwiver

#endif
