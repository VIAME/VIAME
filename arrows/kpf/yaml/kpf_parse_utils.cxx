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
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( "arrows.kpf.kpf_parse_utils" ) );

using std::string;
using std::map;
using std::pair;
using std::make_pair;
using std::vector;
using std::ostringstream;

namespace kwiver {
namespace vital {
namespace kpf {

//
// Given a string which is expected to be a packet header (e.g.
// 'g0', 'meta', 'eval19') separate it into a success flag,
// the tag string, and the integer domain. Return NO_DOMAIN if
// not present (e.g. 'meta')
//

header_parse_t
parse_header( const string& s )
{
  auto bad_parse = std::make_tuple( false, string(), packet_header_t::NO_DOMAIN );
  if (s.empty())
  {
    LOG_ERROR( main_logger, "Packet header: tying to parse empty string?" );
    return bad_parse;
  }

  //
  // packet headers are always of the form [a-Z]+[0-9]*?
  //

  // Example used in comments:
  // string: 'eval123'
  // index:   01234567

  // start parsing at the back

  size_t i=s.size()-1;   // e.g. 6

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


packet_header_t
packet_header_parser( const string& s )
{
  //
  // try to parse the header into a flag / tag / domain
  //

  header_parse_t h = parse_header( s );
  if (! std::get<0>(h) )
  {
    return packet_header_t();
  }

  string tag_str( std::get<1>(h) );
  packet_style style = str2style( tag_str );
  // may be a KV packet
  if ( style == packet_style::INVALID )
  {
    style = packet_style::KV;
  }

  int domain( std::get<2>(h) );
  return packet_header_t( style, domain );
}

} // ...kpf
} // ...vital
} // ...kwiver
