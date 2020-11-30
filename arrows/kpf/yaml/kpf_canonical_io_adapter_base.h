// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Base class for adapters for complex types.
 *
 * For complex types (bounding boxes, activities), there's probably not
 * a one-to-one mapping between the canonical KPF representation and the user's
 * data structure.
 *
 * This non-templated base class holds the packet's "bounce buffer" that
 * adapters use after the packet has been transferred out of the packet buffer
 * as it goes into user-space.
 */

#ifndef KWIVER_VITAL_KPF_CANONICAL_IO_ADAPTER_BASE_H_
#define KWIVER_VITAL_KPF_CANONICAL_IO_ADAPTER_BASE_H_

#include <arrows/kpf/yaml/kpf_bounce_buffer.h>

namespace kwiver {
namespace vital {
namespace kpf {

class kpf_reader_t;

struct kpf_canonical_io_adapter_base
{
  packet_bounce_t packet_bounce;
  kpf_canonical_io_adapter_base& set_domain( int d ) { this->packet_bounce.set_domain(d); return *this; }
};

KPF_YAML_EXPORT
kpf_reader_t& operator>>( kpf_reader_t& t,
                          kpf_canonical_io_adapter_base& io );

} // ...kpf
} // ...vital
} // ...kwiver

#endif
