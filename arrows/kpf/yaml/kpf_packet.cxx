/*ckwg +29
 * Copyright 2017-2018 by Kitware, Inc.
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
 * \brief KPF packet implementation.
 *
 */

#include "kpf_packet.h"
#include <arrows/kpf/yaml/kpf_canonical_types.h>

#include <stdexcept>
#include <sstream>

namespace kwiver {
namespace vital {
namespace kpf {


packet_t
::~packet_t()
{
  if (this->header.style == packet_style::CSET)
  {
    delete this->cset;
  }
}

packet_t
::packet_t( const packet_t& other )
  : header( other.header ), cset(0)
{
  *this = other;
}

packet_t
::packet_t( const packet_header_t& h )
  : header(h), cset(0)
{
  switch (this->header.style)
  {
  case packet_style::INVALID:
    // do nothing
    break;
  case packet_style::ID:
    new (& (this->id)) canonical::id_t();
    break;
  case packet_style::TS:
    new (& (this->timestamp)) canonical::timestamp_t();
    break;
  case packet_style::CONF:
    new (& (this->conf)) canonical::conf_t();
    break;
  case packet_style::CSET:
    new (& (this->cset)) canonical::cset_t*( new canonical::cset_t() );
    break;
  case packet_style::EVAL:
    new (& (this->eval)) canonical::eval_t();
    break;
  case packet_style::TSR:
    new (& (this->timestamp_range)) canonical::timestamp_range_t();
    break;
  case packet_style::KV:
    new (& (this->kv)) canonical::kv_t();
    break;
  case packet_style::GEOM:
    new (& (this->bbox)) canonical::bbox_t();
    break;
  case packet_style::POLY:
    new (& (this->poly)) canonical::poly_t();
    break;
  case packet_style::META:
    new (& (this->meta)) canonical::meta_t();
    break;
  case packet_style::ACT:
    new (& (this->activity)) canonical::activity_t();
    break;

  default:
    {
      std::ostringstream oss;
      oss << "Unhandled ctor for style " << style2str(this->header.style) << " (domain " << this->header.domain << ")";
      throw std::logic_error( oss.str() );
    }
  }
}

packet_t&
packet_t
::operator=( const packet_t& other )
{
  // quick exit on self-assignment
  if (this == &other) return *this;

  // don't leak the cset
  if (this->header.style == packet_style::CSET)
  {
    delete this->cset;
  }

  // copy over the header
  this->header = other.header;

  switch (this->header.style)
  {

  case packet_style::INVALID:
    // do nothing
    break;
  case packet_style::ID:
    new (& (this->id)) canonical::id_t( other.id.d );
    break;
  case packet_style::TS:
    new (& (this->timestamp)) canonical::timestamp_t( other.timestamp.d );
    break;
  case packet_style::CONF:
    new (& (this->conf)) canonical::conf_t( other.timestamp.d );
    break;
  case packet_style::CSET:
    new (& (this->cset)) canonical::cset_t*( new canonical::cset_t( *other.cset) );
    break;
  case packet_style::EVAL:
    new (& (this->eval)) canonical::eval_t( other.eval.d );
    break;
  case packet_style::TSR:
    new (& (this->timestamp_range)) canonical::timestamp_range_t( other.timestamp_range );
    break;
  case packet_style::KV:
    new (& (this->kv)) canonical::kv_t( other.kv );
    break;
  case packet_style::GEOM:
    new (& (this->bbox)) canonical::bbox_t( other.bbox );
    break;
  case packet_style::POLY:
    new (& (this->poly)) canonical::poly_t( other.poly );
    break;
  case packet_style::META:
    new (& (this->meta)) canonical::meta_t( other.meta );
    break;
  case packet_style::ACT:
    new (& (this->activity)) canonical::activity_t( other.activity );
    break;

  default:
    {
      std::ostringstream oss;
      oss << "Unhandled cpctor for style " << style2str(this->header.style) << " (domain " << this->header.domain << ")";
      throw std::logic_error( oss.str() );
    }
  }
  return *this;
}

packet_t
::packet_t( packet_t&& other )
{
  *this = other;
  if (other.header.style == packet_style::CSET)
  {
    other.cset = nullptr;
  }
}

packet_t&
packet_t
::operator=( packet_t&& other )
{
  if (this == &other)
  {
    return *this;
  }
  if (this->header.style == packet_style::CSET)
  {
    delete this->cset;
  }
  *this = other;
  if (this->header.style == packet_style::CSET)
  {
    other.cset = nullptr;
  }
  return *this;
}

std::ostream&
operator<<( std::ostream& os, const packet_header_t& p )
{
  os << style2str(p.style) << "/" << p.domain;
  return os;
}

std::ostream&
operator<<( std::ostream& os, const packet_t& p )
{
  os << p.header << " ; ";
  switch (p.header.style)
  {
  case packet_style::ID:    os << p.id.d; break;
  case packet_style::TS:    os << p.timestamp.d; break;
  case packet_style::TSR:   os << p.timestamp_range.start << ":" << p.timestamp_range.stop; break;
  case packet_style::GEOM:  os << p.bbox.x1 << ", " << p.bbox.y1 << " - " << p.bbox.x2 << ", " << p.bbox.y2; break;
  case packet_style::KV:    os << p.kv.key << " = " << p.kv.val; break;
  case packet_style::POLY:  os << "(polygon w/ " << p.poly.xy.size() << " points)"; break;
  case packet_style::META:  os << "meta: " << p.meta.txt; break;
  case packet_style::CONF:  os << "conf: " << p.conf.d; break;
  case packet_style::EVAL:  os << "eval: " << p.eval.d; break;
  case packet_style::CSET:
    {
      os << "[ ";
      for (auto conf: p.cset->d)
      {
        os << conf.first << ": " << conf.second << ", ";
      }
      os << "] ";
    }
    break;
  case packet_style::ACT:
    {
      const canonical::activity_t& act = p.activity;
      os << " what [ ";
      for (auto t: act.activity_labels.d)
      {
        os << t.first << ": " << t.second << ", ";
      }
      os << " ] id " << act.activity_id.t.d << "/" << act.activity_id.domain
         << " [ ";
      for (auto t: act.timespan )
      {
        os << t.t.start << ":" << t.t.stop << " /" << t.domain << ", ";
      }
      os << "] ; actors: [ ";
      for (auto a: act.actors )
      {
        os << a.actor_id.t.d << "/" << a.actor_id.domain << "(";
        for (auto t: a.actor_timespan)
        {
          os << t.t.start << ":" << t.t.stop << " /" << t.domain << ", ";
        }
        os << "), ";
      }
      for (auto k: act.attributes)
      {
        os << k.key << "=" << k.val << " ,";
      }
    }
    break;

  default:
    os << "(undefined payload output)";
    break;

  }
  return os;

}

} // ...kpf
} // ...vital
} // ...kwiver
