#ifndef KWIVER_VITAL_KPF_CANONICAL_IO_ADAPTER_BASE_H_
#define KWIVER_VITAL_KPF_CANONICAL_IO_ADAPTER_BASE_H_


#include <vital/kpf/vital_kpf_export.h>
#include <vital/kpf/kpf_bounce_buffer.h>

namespace kwiver {
namespace vital {
namespace kpf {

//
// For complex types, such as bounding box, packet_bounce_t
// needs to be associated with functions to convert between
// the KPF and user types; this base class holds the
// packet_bounce_t instance.
//

class kpf_reader_t;

struct kpf_canonical_io_adapter_base
{
  packet_bounce_t packet_bounce;
  kpf_canonical_io_adapter_base& set_domain( int d ) { this->packet_bounce.set_domain(d); return *this; }
};

VITAL_KPF_EXPORT
kpf_reader_t& operator>>( kpf_reader_t& t,
                          kpf_canonical_io_adapter_base& io );

} // ...kpf
} // ...vital
} // ...kwiver

#endif
