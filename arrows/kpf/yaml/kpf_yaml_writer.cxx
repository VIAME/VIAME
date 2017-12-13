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
 * \brief YAML writers.
 *
 * We just write the text directly; it ensures we have inline YAML and also
 * reinforces the concept that KPF-YAML is a subset of YAML.
 *
 */

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
  w.s << io.kv.key << ": " << io.kv.val << ", ";
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
operator<<( record_yaml_writer& w, const writer< canonical::eval_t >& io)
{
  w.ensure_start();
  w.s << "eval" << io.domain << ": " << io.eval.d << ", ";
  return w;
}

record_yaml_writer&
operator<<( record_yaml_writer& w, const writer< canonical::poly_t >& io)
{
  w.ensure_start();
  w.s << "poly" << io.domain << ": [";
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
