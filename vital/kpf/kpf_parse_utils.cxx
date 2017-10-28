#include "kpf_parse_utils.h"

#include <vital/exceptions/kpf.h>

#include <utility>
#include <cctype>
#include <vector>
#include <sstream>
#include <map>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::string;
using std::map;
using std::pair;
using std::make_pair;
using std::vector;
using std::ostringstream;

namespace { // anon

using kwiver::vital::kpf::packet_style;
using kwiver::vital::kpf::packet_t;
using namespace kwiver::vital::kpf::canonical;

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
//

void
need_at_least( const string& tag, size_t needed, size_t index, size_t s )
{
  //
  // if (say) needed == 4, then index range is [index] ... [index+3]
  // need to make sure [index+3] is valid
  // i.e. that last-valid-index (aka s-1) is >= index+3
  // index + needed - 1 <= s-1, or...
  // index + needed <= s
  // if ! (index + needed <=s ), throw
  // ! (index + needed <= s) ==> index + needed > s
  if (index + needed > s )
  {
    ostringstream oss;
    oss << "Parsing " << tag << ": at index " << index << ", " << needed
        << " tokens required but only " << s << " in buffer";
    throw ::kwiver::vital::kpf_token_underrun_exception( oss.str() );
  }
}

pair< bool, size_t >
parse_geom( size_t index,
            const vector<string>& tokens,
            packet_t& packet )
{
  need_at_least( "geom", 4, index, tokens.size());

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

  packet.bbox = bbox_t( xy[0], xy[1], xy[2], xy[3] );
  return make_pair( true, index+4 );
}

pair< bool, size_t >
parse_poly( size_t index,
            const vector<string>& tokens,
            packet_t& packet )
{
  need_at_least( "poly-npts", 1, index, tokens.size() );
  size_t npts = 0;
  try
  {
    npts = stoi( tokens[ index ] );
  }
  catch (const std::invalid_argument& e )
  {
    LOG_ERROR( main_logger, "parsing poly: error converting npoints to int " << e.what() );
    return make_pair( false, index );
  }

  need_at_least( "poly-pts", (npts*2)+1, index, tokens.size() );
  try
  {
    ++index;
    for (size_t i=0; i<npts; ++i)
    {
      double x = stod( tokens[index++] );
      double y = stod( tokens[index++] );
      packet.poly.xy.push_back( make_pair( x, y ));
    }
  }
  catch (const std::invalid_argument& e )
  {
    LOG_ERROR( main_logger, "parsing poly: error converting x/y to double " << e.what() );
    return make_pair( false, index );
  }
  return make_pair( true, index );
}

pair< bool, size_t >
parse_scalar( size_t index,
              const vector<string>& tokens,
              packet_style style,
              packet_t& packet )
{
  need_at_least( style2str(style), 1, index, tokens.size() );

  try
  {
    switch (style)
    {
      case packet_style::ID:
        packet.id.d = stoi( tokens[ index ] );
        break;
      case packet_style::TS:
        packet.timestamp.d = stod( tokens[ index ]);
        break;
      case packet_style::CONF:
        packet.conf.d = stod( tokens[index] );
        break;
      default:
        LOG_ERROR( main_logger, "Unhandled scalar parse style " << static_cast<int>( style ) );
        return make_pair( false, index );
    }
  }
  catch (const std::invalid_argument& e)
  {
    LOG_ERROR( main_logger, "parsing scalar: error converting to scalar " << e.what() );
    return make_pair( false, index );
  }

  return make_pair( true, index+1 );
}

pair< bool, size_t >
parse_kv( size_t index,
          const vector<string>& tokens,
          packet_t& packet )
{
  need_at_least( "kv", 2, index, tokens.size() );

  packet.kv.key = tokens[index];
  packet.kv.val = tokens[index+1];

  return make_pair( true, index+2 );
}

pair< bool, size_t >
parse_meta( size_t index,
            const vector<string>& tokens,
            packet_t& packet )
{
  string s("");
  for (; index < tokens.size(); ++index)
  {
    s += tokens[index];
    if (index != tokens.size()-1)
    {
      s += " ";
    }
  }
  packet.header.domain = kwiver::vital::kpf::packet_header_t::NO_DOMAIN;
  packet.meta = s;
  return make_pair( true, index );
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
  // hope
  //

  auto ret = make_pair( false, size_t() );

  switch (packet.header.style)
  {

  case packet_style::GEOM:
    ret = parse_geom( index, tokens, packet);
    break;

  case packet_style::POLY:
    new (&packet.poly ) canonical::poly_t();
    ret = parse_poly( index, tokens, packet);
    break;

  case packet_style::ID:   // fallthrough
  case packet_style::TS:   // fallthrough
  case packet_style::CONF: // fallthrough
    ret = parse_scalar( index, tokens, packet.header.style, packet );
    break;

  case packet_style::KV:
    new (&packet.kv) canonical::kv_t("", "");
    ret = parse_kv( index, tokens, packet );
    break;

  case packet_style::META:
    new (&packet.meta) canonical::meta_t();
    ret = parse_meta( index, tokens, packet );
    break;

  default:
    LOG_ERROR( main_logger, "Unparsed packet style '" << style2str( packet.header.style) << "'" );
    break;

  }

  return ret;

}



} // ...kpf
} // ...vital
} // ...kwiver
