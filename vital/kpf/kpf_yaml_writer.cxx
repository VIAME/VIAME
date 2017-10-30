#include "kpf_yaml_writer.h"

kwiver::vital::kpf::private_endl_t kwiver::vital::kpf::record_yaml_writer::endl;

namespace kwiver {
namespace vital {
namespace kpf {

void
record_yaml_writer
::ensure_start()
{
  if (! this->line_started )
  {
    this->s << "- { ";
    this->line_started = true;
  }
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const private_endl_t& )
{
  w.ensure_start();
  w.s << " }" << std::endl;
  w.line_started = false;
  return w;
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const writer< canonical::id_t >& io)
{
  w.ensure_start();
  w.s << "id" << io.domain << ": " << io.id.d << ", ";
  return w;
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const writer< canonical::bbox_t >& io)
{
  w.ensure_start();
  w.s << "g" << io.domain << ": " << io.box.x1 << " " << io.box.y1 << " " << io.box.x2 << " " << io.box.y2 << ", ";
  return w;
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const writer< canonical::timestamp_t >& io)
{
  w.ensure_start();
  w.s << "ts" << io.domain << ": " << io.ts.d << ", ";
  return w;
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const writer< canonical::kv_t >& io)
{
  w.ensure_start();
  w.s << "kv: " << io.kv.key << " " << io.kv.val << ", ";
  return w;
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const writer< canonical::conf_t >& io)
{
  w.ensure_start();
  w.s << "conf" << io.domain << ": " << io.conf.d << ", ";
  return w;
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const writer< canonical::poly_t >& io)
{
  w.ensure_start();
  w.s << "poly" << io.domain << ": " << io.poly.xy.size() << " [";
  for (const auto& p : io.poly.xy )
  {
    w.s << "[ " << p.first << ", " << p.second << " ],";
  }
  w.s << "], ";
  return w;
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const writer< canonical::meta_t >& io)
{
  w.ensure_start();
  w.s << "meta: " << io.meta.txt;
  return w;
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const writer< canonical::timestamp_range_t >& io )
{
  w.ensure_start();
  w.s << "tsr" << io.domain << ": [" << io.tsr.start << " , " << io.tsr.stop << "] ,";
  return w;
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const writer< canonical::activity_t >& io )
{
  const canonical::activity_t& act = io.activity;

  w.ensure_start();
  w.s << "act" << io.domain << ": ";
  w.s << act.activity_name << ", ";

  w.s << "id" << act.activity_id_domain << ": " << act.activity_id.d << ", ";

  w.s << "timespan: [{ ";
  for (auto t: act.timespan )
  {
    w.s << "tsr" << t.domain << ": [" << t.tsr.start << " , " << t.tsr.stop << "], ";
  }
  w.s << " }], ";

  for (auto k: act.attributes )
  {
    w.s << k.key << ": " << k.val << ", ";
  }

  w.s << "actors: [{ ";
  for (auto a: act.actors)
  {
    w.s << "id" << a.id_domain << ": " << a.id.d << ", ";
    w.s << "timespan: [{ ";
    for (auto t: a.actor_timespan )
    {
      w.s << "tsr" << t.domain << ": [" << t.tsr.start << " , " << t.tsr.stop << "], ";
    }
    w.s << " }], ";
  }
  w.s << "}], ";

  return w;
}

} // ...kpf
} // ...vital
} // ...kwiver
