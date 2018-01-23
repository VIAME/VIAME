/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
