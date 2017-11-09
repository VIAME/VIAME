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
 * \brief KPF packet implementation.
 *
 */

#include "kpf_packet.h"

#include <stdexcept>
#include <sstream>
#include <map>

using std::map;
using std::string;

namespace { // anon

using kwiver::vital::kpf::packet_style;
using kwiver::vital::kpf::packet_t;
using namespace kwiver::vital::kpf::canonical;

//
// This structure defines the mapping between text tags and
// their corresponding enums.
//

struct tag2type_bimap_t
{
  map< string, packet_style > tag2style;
  map< packet_style, string > style2tag;

  tag2type_bimap_t()
  {
    this->style2tag[ packet_style::INVALID ] = "invalid";
    this->style2tag[ packet_style::ID ] = "id";
    this->style2tag[ packet_style::TS ] = "ts";
    this->style2tag[ packet_style::TSR ] = "tsr";
    this->style2tag[ packet_style::LOC ] = "loc";
    this->style2tag[ packet_style::GEOM ] = "g";
    this->style2tag[ packet_style::POLY ] = "poly";
    this->style2tag[ packet_style::CONF ] = "conf";
    this->style2tag[ packet_style::ACT ] = "act";
    this->style2tag[ packet_style::EVAL ] = "eval";
    this->style2tag[ packet_style::ATTR ] = "a";
    this->style2tag[ packet_style::TAG ] = "tag";
    this->style2tag[ packet_style::KV ] = "kv";
    this->style2tag[ packet_style::META ] = "meta";


    for (auto i=this->style2tag.begin(); i != this->style2tag.end(); ++i )
    {
      this->tag2style[ i->second ] = i->first;

    }
 };
};

static tag2type_bimap_t TAG2TYPE_BIMAP;

} // ...anon


namespace kwiver {
namespace vital {
namespace kpf {

bool
packet_header_cmp
::operator()( const packet_header_t& lhs, const packet_header_t& rhs ) const
{ return ( lhs.style == rhs.style )
  ? (lhs.domain < rhs.domain)
  : (lhs.style < rhs.style);
}


//
// Given a string, return its corresponding packet style
//

packet_style
str2style( const string& s )
{
  auto probe = TAG2TYPE_BIMAP.tag2style.find( s );
  return
    (probe == TAG2TYPE_BIMAP.tag2style.end())
    ? packet_style::INVALID
    : probe->second;

}

//
// Given a style, return its corresponding string
//

string
style2str( packet_style s )
{
  auto probe = TAG2TYPE_BIMAP.style2tag.find( s );
  return
    (probe == TAG2TYPE_BIMAP.style2tag.end())
    ? "invalid"
    : probe->second;
}



packet_t
::~packet_t()
{
  // all payload variants trivially destructable so far
}

packet_t
::packet_t( const packet_t& other ):
  header( other.header )
{
  *this = other;
}

packet_t&
packet_t
::operator=( const packet_t& other )
{
  // quick exit on self-assignment
  if (this == &other) return *this;

  // copy over the header
  this->header = other.header;

  switch (this->header.style)
  {
    // just copy over trivial types
  case packet_style::INVALID: break;
  case packet_style::ID:      this->id = other.id; break;
  case packet_style::TS:      this->timestamp = other.timestamp; break;
  case packet_style::CONF:    this->conf = other.conf; break;

    // placement new for non-trivial types
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

  case packet_style::ACT:
    {
      const activity_t& act = p.activity;
      os << act.activity_name << " id " << act.activity_id.d << "/" << act.activity_id_domain << " [ ";
      for (auto t: act.timespan )
      {
        os << t.tsr.start << ":" << t.tsr.stop << " /" << t.domain << ", ";
      }
      os << "] ; actors: [ ";
      for (auto a: act.actors )
      {
        os << a.id.d << "/" << a.id_domain << "(";
        for (auto t: a.actor_timespan)
        {
          os << t.tsr.start << ":" << t.tsr.stop << " /" << t.domain << ", ";
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
    os << "(unhandled output for " << p << ")";
    break;

  }
  return os;

}

} // ...kpf
} // ...vital
} // ...kwiver

