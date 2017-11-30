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
 * \brief YAML parser.
 *
 * The YAML parser recursively vists each node and tokenizes the scalars,
 * resulting in a flat list of tokens that the event parsers have to
 * reconstruct. Again, probably not optimal.
 *
 */

#include "kpf_yaml_parser.h"

#include <string>
#include <vector>
#include <sstream>
#include <cctype>

#include <vital/util/tokenize.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );


using std::istream;
using std::string;
using std::vector;
using std::isdigit;
using std::ostringstream;

namespace { // anon

/**
 * \brief Visit the node.
 *
 * Extract the content from the node and append to the token buffer.
 *
 */

void visit( const YAML::Node& n, vector< string >& tokens )
{
  switch (n.Type())
  {
  case YAML::NodeType::Scalar:
    ::kwiver::vital::tokenize( n.as<string>(), tokens, " ", true );
    break;

  case YAML::NodeType::Sequence:
    for (auto it=n.begin(); it != n.end(); ++it)
    {
      visit( *it, tokens );
    }
    break;

  case YAML::NodeType::Map:
    for (auto it=n.begin(); it != n.end(); ++it)
    {
      visit( it->first, tokens );
      visit( it->second, tokens );
    }
    break;

  case YAML::NodeType::Null:  // fall-through
  case YAML::NodeType::Undefined:
    // do nothing
    break;
  }
}

} // ...anon

namespace kwiver {
namespace vital {
namespace kpf {

/**
 * \brief Load the YAML document in the constructor.
 *
 * The entire document is loaded here; each "line" can be
 * accessed by iterating over the root.
 *
 */

kpf_yaml_parser_t
::kpf_yaml_parser_t( istream& is )
{
  try
  {
    this->root = YAML::Load( is );
  }
  // This seems not to work on OSX as of 30oct2017
  // see https://stackoverflow.com/questions/21737201/problems-throwing-and-catching-exceptions-on-os-x-with-fno-rtti
  catch (const YAML::ParserException& e )
  {
    LOG_ERROR( main_logger, "Exception parsing KPF YAML: " << e.what() );
    this->root = YAML::Node();
  }
  this->current_record = this->root.begin();
}

bool
kpf_yaml_parser_t
::get_status() const
{
  return this->current_record != this->root.end();
}

/**
 * \brief Read the next child of root into the packet buffer.
 *
 */

bool
kpf_yaml_parser_t
::parse_next_record( packet_buffer_t& local_packet_buffer )
{
  if (this->current_record == this->root.end())
  {
    return false;
  }
  const YAML::Node& n = *(this->current_record++);
  vector< string > tokens;

  if (! n.IsMap())
  {
    LOG_ERROR( main_logger, "YAML: type " << n.Type() << " not a map? " << n.as<string>() );
    return false;
  }

  // start parsing the map entries; each must be a packet header
  for (auto it=n.begin(); it != n.end(); ++it)
  {
    string s = it->first.as<string>();
    // if the last character is not a digit, it's a key (or a meta)
    bool is_meta = (s == "meta");
    bool ends_in_digit =  (!s.empty()) && ( isdigit( static_cast< unsigned char>( s.back() )));
    if (is_meta || ends_in_digit)
    {
      // it's a KPF packet
      tokens.push_back( s+":" );
    }
    else
    {
      // it's a KPF 'kv:' key-value pair
      tokens.push_back( "kv:" );
      tokens.push_back( s );
    }

    //
    // MASSIVE hack for polygons-- the (token-based) parse_poly() routine
    // requires the number of points in the token stream, but that's not
    // explicitly stored in the YAML. Now that we've abandoned non-yaml
    // text packets, we should fix this, but for now...
    //
    bool is_poly = (s.substr(0, 4) == "poly");
    if (is_poly)
    {
      vector<string> poly_tokens;
      visit( it->second, poly_tokens );
      ostringstream oss;
      oss << (poly_tokens.size() / 2);
      tokens.push_back( oss.str() );
      tokens.insert( tokens.end(), poly_tokens.begin(), poly_tokens.end());
    }
    else
    {
      visit( it->second, tokens );
    }
  }

  return packet_parser( tokens, local_packet_buffer );
}

} // ...kpf
} // ...vital
} // ...kwiver

