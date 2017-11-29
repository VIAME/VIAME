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
 * \brief KPF parsing utilities.
 *
 * This is where text tokens get converted into KPF canonical structures.
 *
 */

#include "kpf_parse_utils.h"

#include <arrows/kpf/yaml/kpf_exception.h>

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

const string&
next_token( size_t& index, const string& tag, const vector<string>& tokens )
{
  if (index == tokens.size())
  {
    ostringstream oss;
    oss << "Parsing " << tag << ": index " << index << " is at end of token buffer";
    throw ::kwiver::vital::kpf_token_underrun_exception( oss.str() );
  }
  return tokens[index++];
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
    for (auto i=0; i<4; ++i)
    {
      LOG_ERROR( main_logger, "index " << index << " offset " << i << "'" << tokens[ index+i ] << "'" );
    }
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
parse_tsr( size_t index,
           const vector<string>& tokens,
           packet_t& packet )
{
  need_at_least( "tsr", 2, index, tokens.size() );
  try
  {
    packet.timestamp_range.start = stod( tokens[index++] );
    packet.timestamp_range.stop = stod( tokens[index++] );
  }
  catch (const std::invalid_argument& e )
  {
    LOG_ERROR( main_logger, "parsing tsr: error converting start / stop to double " << e.what() );
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

bool
parse_activity_timespan( size_t& index,
                         const vector<string>& tokens,
                         vector< activity_t::scoped_tsr_t >& tsr_list )
{
  //
  // Due to the vagaries of YAML conversion into a flat token stream,
  // a yaml fragment
  //
  //   timespan: [{tsr0: [0, 1526], tsr1: [314, 419]}], src: truth,
  //
  // ends up like this:
  //
  // token: 4: 'kv:'
  // token: 5: 'timespan'
  // token: 6: 'tsr0'
  // token: 7: '0'
  // token: 8: '1526'
  // token: 9: 'tsr1'
  // token: 10: '314'
  // token: 11: '419'
  // token: 12: 'kv:'
  // token: 13: 'src'
  //
  // ...and we're here with index == 7.
  //
  // So: we need to back up, and dump scoped TSR entries into the tsr_list
  // until we see a non-TSR token (indicating we're out of the timespan.)

  --index;

  while ( (index != tokens.size()) &&
          (tokens[index].substr(0, 3) == "tsr"))
  {
    need_at_least( "activity-tsr", 2, index, tokens.size() );
    activity_t::scoped_tsr_t t;
    try
    {
      t.domain = stoi( tokens[index++].substr(3) );
      t.tsr.start = stod( tokens[index++] );
      t.tsr.stop = stod( tokens[index++] );
    }
    catch (const std::invalid_argument& e)
    {
      LOG_ERROR( main_logger, "Parsing activity TSR: couldn't convert " << e.what() );
      return false;
    }
    tsr_list.push_back( t );
  }

  return true;
}

bool
parse_activity_actors( size_t& index,
                       const vector<string>& tokens,
                       vector< activity_t::actor_t >& actor_list )
{
  //
  // Similar to parse_activity_timespan, a YAML fragment such as
  //
  //   actors: [{id1: 7, timespan: [{tsr0: [0, 1526], tsr1: [315, 777]}]} , {id1: 6, timespan: [{tsr0: [0, 1526], tsr1: [888,999]}]} ,  ]}
  //
  // ...flattens out to:
  //
  // token: 15: 'kv:'
  // token: 16: 'actors'
  // token: 17: 'id1'
  // token: 18: '7'
  // token: 19: 'timespan'
  // token: 20: 'tsr0'
  // token: 21: '0'
  // token: 22: '1526'
  // token: 23: 'tsr1'
  // token: 24: '315'
  // token: 25: '777'
  // token: 26: 'id1'
  // token: 27: '6'
  // token: 28: 'timespan'
  // token: 29: 'tsr0'
  // token: 30: '0'
  // token: 31: '1526'
  // token: 32: 'tsr1'
  // token: 33: '888'
  // token: 34: '999'
  //
  // ...and we're here with an index of 18. So back up, parse the id,
  // parse the timespan, until either (a) we're done or (b) next token
  // is neither id nor timespan.
  //

  --index;

  while ( (index != tokens.size()) &&
          (tokens[index].substr(0, 2) == "id"))
  {
    need_at_least( "activity-actor-id", 1, index, tokens.size() );
    activity_t::actor_t actor;
    try
    {
      actor.id_domain = stoi( tokens[index++].substr(2) );
      actor.id = kwiver::vital::kpf::canonical::id_t( stoi( tokens[index++] ));
    }
    catch (const std::invalid_argument& e)
    {
      LOG_ERROR( main_logger, "Parsing activity actor ID: couldn't convert " << e.what() );
      return false;
    }
    need_at_least( "activity-actor-timespan", 2, index, tokens.size() );
    if (tokens[index] != "timespan")
    {
      LOG_ERROR( main_logger, "Parsing activity actor ID: actor "
                 << actor.id.d << "/" << actor.id_domain << " has no timespan?" );
      return false;
    }
    ++index; // advance to 'tsrN'
    ++index; // advance to start time
    if (! parse_activity_timespan( index, tokens, actor.actor_timespan ))
    {
      return false;
    }

    actor_list.push_back( actor );
  }

  return true;
}

pair< bool, size_t >
parse_activity( size_t index,
                const vector<string>& tokens,
                packet_t& packet )
{
  packet.activity.activity_name = next_token( index, "activity name", tokens );

  // hmm, this is a little awkward
  // last packet must be "actors:" kv
  bool keep_going = (index < tokens.size());
  bool packet_okay = false;
  while (keep_going)
  {
    const string& s = tokens[index++];
    if (s.back() != ':')
    {
      LOG_ERROR( main_logger, "Unexpected token '" << s << "' at index " << index
                 << " while parsing activity " << packet.activity.activity_name );
      keep_going = false;
    }
    kwiver::vital::kpf::packet_header_t h;
    keep_going = packet_header_parser( s, h, true );
    if (! keep_going) continue;  // philosophical

    kwiver::vital::kpf::packet_t p(h);
    auto sub_packet_parse = packet_payload_parser( index, tokens, p );
    if (! sub_packet_parse.first )
    {
      keep_going = false;
      continue;
    }
    index = sub_packet_parse.second;

    // special key/value pairs: 'timespan' and 'actors'
    if ( (p.header.style == packet_style::KV) && (p.kv.key == "timespan"))
    {
      keep_going = parse_activity_timespan( index, tokens, packet.activity.timespan );
    }
    else if ( (p.header.style == packet_style::KV) && (p.kv.key == "actors"))
    {
      keep_going = parse_activity_actors( index, tokens, packet.activity.actors );
      if (keep_going)
      {
        packet_okay = true;
        keep_going = false;
      }
    }
    else if (p.header.style == packet_style::ID)
    {
      // explicitly look for the activity ID
      packet.activity.activity_id = p.id;
      packet.activity.activity_id_domain = p.header.domain;
    }
    else if (p.header.style == packet_style::KV)
    {
      // misc. other attributes
      packet.activity.attributes.push_back( p.kv );
    }
    else
    {
      LOG_ERROR( main_logger, "Unexpected packet " << p << " in activity " << packet.activity.activity_name );
      keep_going = false;
    }
  } // ...while keep_going

  return make_pair( packet_okay, index );
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

  case packet_style::TSR:
    new (&packet.timestamp_range) canonical::timestamp_range_t(0, 0);
    ret = parse_tsr( index, tokens, packet );
    break;

  case packet_style::META:
    new (&packet.meta) canonical::meta_t();
    ret = parse_meta( index, tokens, packet );
    break;

  case packet_style::ACT:
    new (&packet.activity) canonical::activity_t();
    ret = parse_activity( index, tokens, packet );
    break;

  default:
    LOG_ERROR( main_logger, "Unparsed packet style '" << style2str( packet.header.style) << "'" );
    break;

  }

  return ret;

}

bool
packet_header_parser( const string& s,
                      packet_header_t& packet_header,
                      bool expect_colon )
{
  //
  // try to parse the header into a flag / tag / domain
  //

  header_parse_t h = parse_header( s, expect_colon );
  if (! std::get<0>(h) )
  {
    return false;
  }

  string tag_str( std::get<1>(h) );
  packet_style style = str2style( tag_str );
  if ( style == packet_style::INVALID )
  {
    //    LOG_ERROR( main_logger, "Bad packet style '" << tag_str << "'" );
    return false;
  }

  int domain( std::get<2>(h) );
  packet_header = packet_header_t( style, domain );
  return true;
}

bool
packet_parser( const vector<string>& tokens,
               packet_buffer_t& packet_buffer )
{
  size_t index(0), n( tokens.size() );
  while ( index < n )
  {
    packet_t p;
    if (packet_header_parser( tokens[ index ],
                              p.header,
                              true ))
    {
      // uh-oh, we couldn't parse it; build up an 'unparsed' key-value packet
      // until we get a parse
      ++index;
      pair< bool, size_t > next = packet_payload_parser( index, tokens, p );
      if (! next.first )
      {
        // This indicates a malformed packet error
        return false;
      }
      index = next.second;

      packet_buffer.insert( make_pair( p.header, p ));
    }
    else
    {
      // uh-oh, we couldn't recognize the header-- build up an 'unparsed' key-value
      // packet
      string unparsed_txt = tokens[index];
      // keep trying until we either parse a header or run out of tokens
      ++index;
      bool keep_going = (index < n);
      while (keep_going)
      {
        if (packet_header_parser( tokens[index], p.header, true ))
        {
          // we found a parsable header-- all done
          keep_going = false;
        }
        else
        {
          unparsed_txt += " "+tokens[index++];
          keep_going = (index < n);
        }
      }

      packet_header_t unparsed_header( packet_style::KV );
      packet_t unparsed( unparsed_header );
      new (&unparsed.kv) canonical::kv_t( "unparsed", unparsed_txt );
      packet_buffer.insert( make_pair( unparsed.header, unparsed ));
    }
  }
  return true;
}

} // ...kpf
} // ...vital
} // ...kwiver
