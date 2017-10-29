#include "kpf_text_writer.h"

kwiver::vital::kpf::private_endl_t kwiver::vital::kpf::record_text_writer::endl;

namespace kwiver {
namespace vital {
namespace kpf {

record_text_writer&
operator<<( record_text_writer& w, const private_endl_t& )
{
  w.s << std::endl;
  return w;
}

record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::id_t >& io)
{
  w.s << "id" << io.domain << ": " << io.id.d << " ";
  return w;
}

record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::bbox_t >& io)
{
  w.s << "g" << io.domain << ": " << io.box.x1 << " " << io.box.y1 << " " << io.box.x2 << " " << io.box.y2 << " ";
  return w;
}

record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::timestamp_t >& io)
{
  w.s << "ts" << io.domain << ": " << io.ts.d << " ";
  return w;
}

record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::kv_t >& io)
{
  w.s << "kv: " << io.kv.key << " " << io.kv.val << " ";
  return w;
}

record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::conf_t >& io)
{
  w.s << "conf" << io.domain << ": " << io.conf.d << " ";
  return w;
}

record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::poly_t >& io)
{
  w.s << "poly" << io.domain << ": " << io.poly.xy.size() << " ";
  for (const auto& p : io.poly.xy )
  {
    w.s << p.first << " " << p.second << " ";
  }
  return w;
}

record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::meta_t >& io)
{
  w.s << "meta: " << io.meta.txt;
  return w;
}


} // ...kpf
} // ...vital
} // ...kwiver
