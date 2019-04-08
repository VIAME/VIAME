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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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

#include "kpf_packet_header.h"

#include <string>
#include <map>

using std::map;
using std::string;

namespace { // anon

using kwiver::vital::kpf::packet_style;

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
    this->style2tag[ packet_style::CSET ] = "cset";
    this->style2tag[ packet_style::ACT ] = "act";
    this->style2tag[ packet_style::EVAL ] = "eval";
    this->style2tag[ packet_style::ATTR ] = "a";
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
operator==( const packet_header_t& lhs, const packet_header_t& rhs )
{
  return ( (lhs.style == rhs.style) && (lhs.domain == rhs.domain) );
}

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

} // ...kpf
} // ...vital
} // ...kwiver
