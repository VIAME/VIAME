#include "kpf_parse_utils.h"

#include <utility>
#include <cctype>
#include <vector>
#include <map>
#include <stdexcept>


#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::string;
using std::map;
using std::pair;
using std::make_pair;
using std::vector;


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
    this->style2tag[ packet_style::EVENT ] = "e";
    this->style2tag[ packet_style::EVAL ] = "eval";
    this->style2tag[ packet_style::ATTR ] = "a";
    this->style2tag[ packet_style::TAG ] = "tag";
    this->style2tag[ packet_style::KV ] = "kv";


    for (auto i=this->style2tag.begin(); i != this->style2tag.end(); ++i )
    {
      this->tag2style[ i->second ] = i->first;

    }
 };
};

static tag2type_bimap_t TAG2TYPE_BIMAP;

//
// parse the geometry / bounding box:
// 'g0: x1 y1 x2 y2'
//
// example: g0: x1 y1 x2 y2
// index:   0   1   2  3  4
//
// if tokens is:
// ....   g0: x1 y1 x2 y2 g1: ...
// index   8  9  10 11 12 13  ...
//
// ...then for 'g0:' (tokens[8]), index will be 9

pair< bool, size_t >
parse_geom( size_t index,
            const vector<string>& tokens,
            packet_t& packet )
{
  // do we have at least four tokens?
  // tokens[index] is first, through tokens[index+3]
  if (index + 3 >= tokens.size())
  {
    LOG_ERROR( main_logger, "parsing geom: index " << index << " but only " << tokens.size() << " tokens left" );
    return make_pair( false, index );
  }

  double xy[4];
  try
  {
    for (auto i=0; i<4; ++i)
    {
      xy[i] = stod( tokens[ index+i ] );
    }
  }
  catch (const std::invalid_argument& e)
  {
    LOG_ERROR( main_logger, "parsing geom: error converting to double " << e.what() );
    return make_pair( false, index );
  }

  packet.payload.bbox = bbox_t( xy[0], xy[1], xy[2], xy[3] );
  return make_pair( true, index+4 );
}


} // anon


namespace kwiver {
namespace vital {
namespace kpf {


//
// Given a string which is expected to be a packet header (e.g.
// 'g0:', 'meta:', 'eval19:') separate it into a success flag,
// the tag string, and the integer domain. Return NO_DOMAIN if
// not present (e.g. 'meta:')
//

header_parse_t
parse_header(const string& s, bool expect_colon )
{
  auto bad_parse = std::make_tuple( false, string(), packet_header_t::NO_DOMAIN );
  if (s.empty())
  {
    LOG_ERROR( main_logger, "Packet header: tying to parse empty string?" );
    return bad_parse;
  }
  if (expect_colon)
  {
    if (s.size() == 1)
    {
      LOG_ERROR( main_logger, "Packet header: invalid packet '" << s << "'" );
      return bad_parse;
    }
  }


  //
  // packet headers are always of the form [a-Z]+[0-9]*:?
  //

  // Example used in comments:
  // string: 'eval123:'
  // index:   01234567

  // start parsing at the back

  size_t i=s.size()-1;   // e.g. 7
  if (expect_colon)
  {
    if (s[i--] != ':')
    {
      LOG_ERROR( main_logger, "Packet header '" << s << "': no trailing colon" );
      return bad_parse;
    }
  }
  // e.g. i is now 6, s[i]=='3'

  //
  // look for the domain
  //
  int domain = packet_header_t::NO_DOMAIN;

  size_t domain_end(i);   // e.g. 6
  while ((i != 0) && std::isdigit( s[i] ))
  {
    --i;
  }
  // e.g. i is now 3, 'l'
  // if we've backed up to the front of the string and it's still digits,
  // which is ill-formed
  if ((i == 0) && std::isdigit( s[i] ))
  {
    LOG_ERROR( main_logger, "Packet header '" << s << "': no packet style");
    return bad_parse;
  }
  size_t domain_start = i+1;   // e.g. domain_start is 4
  if (domain_start <= domain_end)  // when no domain, start > end
  {
    // substr from index 4, length (6-4+1)==3: '123'
    domain = std::stoi( s.substr( domain_start, domain_end - domain_start+1 ));
  }

  //
  // packet style is everything else
  //

  string style = s.substr( 0, i+1 );  // e.g start at index 0, length 4: 'eval'

  return std::make_tuple( true, style, domain);
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
// Having established the packet style (and domain, although only
// the style is needed for parsing), parse the style-specific payload
// from the token stream and convert it into the appropriate canonical
// type.
//

pair< bool, size_t >
packet_payload_parser ( size_t index,
                        const vector< string >& tokens,
                        packet_t& packet )
{
  //
  // tokens[index] is the start of the token stream which we
  // hope to interpret per the style of packet.header
  //

  auto ret = make_pair( false, size_t() );

  switch (packet.header.style)
  {

  case packet_style::GEOM:
    ret = parse_geom( index, tokens, packet);
    break;

  }

  return ret;

}



} // ...kpf
} // ...vital
} // ...kwiver
