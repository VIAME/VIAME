// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
