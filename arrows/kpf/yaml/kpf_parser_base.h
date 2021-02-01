// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file
 * @brief Base class for KPF parsers for various formats.
 *
 * Concrete format-specific instances of this class can be passed to
 * the kpf_reader_t to loop over KPF input.
 *
 */

#ifndef KWIVER_VITAL_KPF_PARSER_BASE_
#define KWIVER_VITAL_KPF_PARSER_BASE_

#include <arrows/kpf/yaml/kpf_parse_utils.h>
#include <arrows/kpf/yaml/kpf_yaml_schemas.h>

namespace kwiver {
namespace vital {
namespace kpf {

class kpf_parser_base_t
{
public:
  kpf_parser_base_t() {}
  virtual ~kpf_parser_base_t() {}

  virtual bool get_status() const = 0;
  virtual bool parse_next_record( packet_buffer_t& ) = 0;
  virtual schema_style get_current_record_schema() const = 0;
  virtual bool eof() const = 0;
};

} // ...kpf
} // ...vital
} // ...kwiver

#endif
