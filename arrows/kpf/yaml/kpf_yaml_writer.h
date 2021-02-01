// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file
 * @brief Output operators for KPF canonical types in YAML.
 *
 * This file writes KPF canonical types to a stream in a YAML-compatible format.
 */

#ifndef KWIVER_VITAL_KPF_YAML_WRITER_H_
#define KWIVER_VITAL_KPF_YAML_WRITER_H_

#include <iostream>
#include <arrows/kpf/yaml/kpf_canonical_types.h>
#include <arrows/kpf/yaml/kpf_canonical_io.h>
#include <arrows/kpf/yaml/kpf_yaml_schemas.h>

namespace kwiver {
namespace vital {
namespace kpf {

struct KPF_YAML_EXPORT private_endl_t
{};

class KPF_YAML_EXPORT record_yaml_writer
{
public:
  explicit record_yaml_writer( std::ostream& os ) : s( os ), line_started(false), schema( schema_style::UNSPECIFIED ), has_meta( false ) {}

  friend KPF_YAML_EXPORT record_yaml_writer& operator<<( record_yaml_writer& w, const packet_t& p );
  friend KPF_YAML_EXPORT record_yaml_writer& operator<<( record_yaml_writer& w, const writer< canonical::id_t >& io );
  friend KPF_YAML_EXPORT record_yaml_writer& operator<<( record_yaml_writer& w, const writer< canonical::bbox_t >& io );
  friend KPF_YAML_EXPORT record_yaml_writer& operator<<( record_yaml_writer& w, const writer< canonical::timestamp_t >& io );
  friend KPF_YAML_EXPORT record_yaml_writer& operator<<( record_yaml_writer& w, const writer< canonical::kv_t >& io );
  friend KPF_YAML_EXPORT record_yaml_writer& operator<<( record_yaml_writer& w, const writer< canonical::conf_t >& io );
  friend KPF_YAML_EXPORT record_yaml_writer& operator<<( record_yaml_writer& w, const writer< canonical::cset_t >& io );
  friend KPF_YAML_EXPORT record_yaml_writer& operator<<( record_yaml_writer& w, const writer< canonical::cset_t >& io );
  friend KPF_YAML_EXPORT record_yaml_writer& operator<<( record_yaml_writer& w, const writer< canonical::eval_t >& io );
  friend KPF_YAML_EXPORT record_yaml_writer& operator<<( record_yaml_writer& w, const writer< canonical::poly_t >& io );
  friend KPF_YAML_EXPORT record_yaml_writer& operator<<( record_yaml_writer& w, const writer< canonical::meta_t >& io );
  friend KPF_YAML_EXPORT record_yaml_writer& operator<<( record_yaml_writer& w, const writer< canonical::timestamp_range_t >& io );
  friend KPF_YAML_EXPORT record_yaml_writer& operator<<( record_yaml_writer& w, const writer< canonical::activity_t >& io );
  friend KPF_YAML_EXPORT record_yaml_writer& operator<<( record_yaml_writer& w, const private_endl_t& );

  static private_endl_t endl;

  record_yaml_writer& set_schema( schema_style s );

private:
  void reset();
  std::ostream& s;
  std::ostringstream oss;
  bool line_started;
  schema_style schema;
  bool has_meta;
};

KPF_YAML_EXPORT
record_yaml_writer& operator<<( record_yaml_writer& w, const packet_t& p );

KPF_YAML_EXPORT
record_yaml_writer& operator<<( record_yaml_writer& w, const writer< canonical::id_t >& io );

KPF_YAML_EXPORT
packet_bounce_t& operator>>( packet_bounce_t& w, const writer< canonical::id_t >& io );

KPF_YAML_EXPORT
record_yaml_writer& operator<<( record_yaml_writer& w, const writer< canonical::bbox_t >& io );

KPF_YAML_EXPORT
record_yaml_writer& operator<<( record_yaml_writer& w, const writer< canonical::timestamp_t >& io );

KPF_YAML_EXPORT
record_yaml_writer& operator<<( record_yaml_writer& w, const writer< canonical::kv_t >& io );

KPF_YAML_EXPORT
record_yaml_writer& operator<<( record_yaml_writer& w, const writer< canonical::conf_t >& io );

KPF_YAML_EXPORT
record_yaml_writer& operator<<( record_yaml_writer& w, const writer< canonical::eval_t >& io );

KPF_YAML_EXPORT
record_yaml_writer& operator<<( record_yaml_writer& w, const writer< canonical::poly_t >& io );

KPF_YAML_EXPORT
record_yaml_writer& operator<<( record_yaml_writer& w, const writer< canonical::meta_t >& io );

KPF_YAML_EXPORT
record_yaml_writer& operator<<( record_yaml_writer& w, const writer< canonical::timestamp_range_t >& io );

KPF_YAML_EXPORT
record_yaml_writer& operator<<( record_yaml_writer& w, const writer< canonical::activity_t >& io );

KPF_YAML_EXPORT
record_yaml_writer& operator<<( record_yaml_writer& w, const private_endl_t& e );

} // ...kpf
} // ...vital
} // ...kwiver

#endif
