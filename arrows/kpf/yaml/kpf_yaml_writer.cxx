// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief YAML writers.
 *
 * We just write the text directly; it ensures we have inline YAML and also
 * reinforces the concept that KPF-YAML is a subset of YAML.
 *
 */

#include "kpf_yaml_writer.h"
#include <arrows/kpf/yaml/kpf_yaml_schemas.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( "arrows.kpf.kpf_yaml_writer" ) );

#include <ios>

kwiver::vital::kpf::private_endl_t kwiver::vital::kpf::record_yaml_writer::endl;

namespace kwiver {
namespace vital {
namespace kpf {

record_yaml_writer&
record_yaml_writer
::set_schema( schema_style new_schema )
{
  if (this->line_started)
  {
    LOG_ERROR( main_logger, "KPF yaml writer: can't change schemas mid-stream; was "
               << validation_data::schema_style_to_str( this->schema )
               << "; attempted to change to "
               << validation_data::schema_style_to_str( new_schema ) );
    this->s.setstate( std::ios::failbit );
  }
  else
  {
    this->schema = new_schema;
  }
  return *this;
}

void
record_yaml_writer
::reset()
{
  this->line_started = false;
  this->schema = schema_style::UNSPECIFIED;
  this->has_meta = false;
  this->oss.str("");
  this->oss.clear();
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const private_endl_t& )
{
  if (w.has_meta)
  {
    w.s << "- { " << w.oss.str() << " }" << std::endl;
  }
  else
  {
    w.s << "- { " << validation_data::schema_style_to_str( w.schema )
        << ": { " << w.oss.str() << " } }" << std::endl;
  }
  w.reset();
  return w;
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const writer< canonical::id_t >& io)
{
  w.oss << "id" << io.domain << ": " << io.id.d << ", ";
  return w;
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const writer< canonical::bbox_t >& io)
{
  w.oss << "g" << io.domain << ": " << io.box.x1 << " " << io.box.y1 << " " << io.box.x2 << " " << io.box.y2 << ", ";
  return w;
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const writer< canonical::timestamp_t >& io)
{
  w.oss << "ts" << io.domain << ": " << io.ts.d << ", ";
  return w;
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const writer< canonical::kv_t >& io)
{
  w.oss << io.kv.key << ": " << io.kv.val << ", ";
  return w;
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const writer< canonical::conf_t >& io)
{
  w.oss << "conf" << io.domain << ": " << io.conf.d << ", ";
  return w;
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const writer< canonical::cset_t >& io)
{
  w.oss << "cset" << io.domain << ": {";
  for (auto p: io.cset.d )
  {
    w.oss << p.first << ": " << p.second << ", ";
  }
  w.oss << "}, ";
  return w;
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const writer< canonical::eval_t >& io)
{
  w.oss << "eval" << io.domain << ": " << io.eval.d << ", ";
  return w;
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const writer< canonical::poly_t >& io)
{
  w.oss << "poly" << io.domain << ": [";
  for (const auto& p : io.poly.xy )
  {
    w.oss << "[ " << p.first << ", " << p.second << " ],";
  }
  w.oss << "], ";
  return w;
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const writer< canonical::meta_t >& io)
{
  w.oss << "meta: " << io.meta.txt;
  w.has_meta = true;
  return w;
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const writer< canonical::timestamp_range_t >& io )
{
  w.oss << "tsr" << io.domain << ": [" << io.tsr.start << " , " << io.tsr.stop << "] ,";
  return w;
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const writer< canonical::activity_t >& io )
{
  const canonical::activity_t& act = io.activity;

  w.oss << "act" << io.domain << ": ";
  w.oss << "{ ";
  for (auto p:act.activity_labels.d)
  {
    w.oss << p.first << ": " << p.second << ", ";
  }
  w.oss << "}, ";

  w.oss << "id" << act.activity_id.domain << ": " << act.activity_id.t.d << ", ";

  w.oss << "timespan: [{ ";
  for (auto t: act.timespan )
  {
    w.oss << "tsr" << t.domain << ": [" << t.t.start << " , " << t.t.stop << "], ";
  }
  w.oss << " }], ";

  for (auto k: act.attributes )
  {
    w << writer<canonical::kv_t>( k );
  }

  for (auto e: act.evals )
  {
    w << writer< canonical::eval_t>( e );
  }

  w.oss << "actors: [ ";
  for (auto a: act.actors)
  {
    w.oss << "{ ";
    w.oss << "id" << a.actor_id.domain << ": " << a.actor_id.t.d << ", ";
    w.oss << "timespan: [{ ";
    for (auto t: a.actor_timespan )
    {
      w.oss << "tsr" << t.domain << ": [" << t.t.start << " , " << t.t.stop << "], ";
    }
    w.oss << " }], ";
    w.oss << " }, ";
  }
  w.oss << "], ";

  return w;
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const packet_t& p )
{
  auto d = p.header.domain;
  switch (p.header.style)
  {
  case packet_style::META:   w << writer< canonical::meta_t>( p.meta.txt ); break;
  case packet_style::ID:     w << writer< canonical::id_t>( p.id, d ); break;
  case packet_style::TS:     w << writer< canonical::timestamp_t>( p.timestamp, d );  break;
  case packet_style::TSR:    w << writer< canonical::timestamp_range_t>( p.timestamp_range, d); break;
  case packet_style::GEOM:   w << writer< canonical::bbox_t>( p.bbox, d ); break;
  case packet_style::POLY:   w << writer< canonical::poly_t>( p.poly, d ); break;
  case packet_style::CONF:   w << writer< canonical::conf_t>( p.conf, d ); break;
  case packet_style::CSET:   w << writer< canonical::cset_t>( *p.cset, d ); break;
  case packet_style::ACT:    w << writer< canonical::activity_t>( p.activity, d ); break;
  case packet_style::EVAL:   w << writer< canonical::eval_t>( p.eval, d ); break;
  case packet_style::KV:     w << writer< canonical::kv_t>( p.kv ); break;
  default:
    LOG_ERROR( main_logger, "No KPF packet writer for " << p );
    break;
  }
  return w;
}

} // ...kpf
} // ...vital
} // ...kwiver
