#ifndef KWIVER_VITAL_KPF_TEXT_WRITER_H_
#define KWIVER_VITAL_KPF_TEXT_WRITER_H_

#include <vital/kpf/vital_kpf_export.h>

#include <iostream>
#include <vital/kpf/kpf_canonical_types.h>
#include <vital/kpf/kpf_canonical_io.h>

namespace kwiver {
namespace vital {
namespace kpf {

struct VITAL_KPF_EXPORT private_endl_t
{};

class VITAL_KPF_EXPORT record_text_writer
{
public:
  explicit record_text_writer( std::ostream& os ) : s( os ) {}

  friend record_text_writer& operator<<( record_text_writer& w, const writer< canonical::id_t >& io );
  friend record_text_writer& operator<<( record_text_writer& w, const writer< canonical::bbox_t >& io );
  friend record_text_writer& operator<<( record_text_writer& w, const writer< canonical::timestamp_t >& io );
  friend record_text_writer& operator<<( record_text_writer& w, const writer< canonical::kv_t >& io );
  friend record_text_writer& operator<<( record_text_writer& w, const writer< canonical::conf_t >& io );
  friend record_text_writer& operator<<( record_text_writer& w, const writer< canonical::poly_t >& io );
  friend record_text_writer& operator<<( record_text_writer& w, const writer< canonical::meta_t >& io );
  friend record_text_writer& operator<<( record_text_writer& w, const private_endl_t& );

  static private_endl_t endl;

private:
  std::ostream& s;
};

VITAL_KPF_EXPORT
record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::id_t >& io );

VITAL_KPF_EXPORT
packet_bounce_t&
operator>>( packet_bounce_t& w, const writer< canonical::id_t >& io );

VITAL_KPF_EXPORT
record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::bbox_t >& io );

VITAL_KPF_EXPORT
record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::timestamp_t >& io );

VITAL_KPF_EXPORT
record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::kv_t >& io );

VITAL_KPF_EXPORT
record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::conf_t >& io );

VITAL_KPF_EXPORT
record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::poly_t >& io );

VITAL_KPF_EXPORT
record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::meta_t >& io );

VITAL_KPF_EXPORT
record_text_writer&
operator<<( record_text_writer& w, const private_endl_t& e );


} // ...kpf
} // ...vital
} // ...kwiver


#endif
