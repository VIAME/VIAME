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
 * \brief YAML parser.
 *
 * The YAML parser recursively vists each node and tokenizes the scalars,
 * resulting in a flat list of tokens that the event parsers have to
 * reconstruct. Again, probably not optimal.
 *
 */

#include "kpf_yaml_parser.h"

#include <arrows/kpf/yaml/kpf_yaml_schemas.h>

#include <string>
#include <vector>
#include <sstream>
#include <cctype>

#include <vital/util/tokenize.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( "arrows.kpf.kpf_yaml_parser" ) );


using std::istream;
using std::string;
using std::vector;
using std::pair;
using std::isdigit;
using std::ostringstream;
using std::make_pair;
using std::stod;

namespace { // anon

namespace KPF = ::kwiver::vital::kpf;
namespace KPFC = ::kwiver::vital::kpf::canonical;

KPF::schema_style
kpf_v3_check( const YAML::Node& n )
{
  if ( ! n.IsMap() )    return KPF::schema_style::INVALID;
  if ( n.size() != 1)   return KPF::schema_style::INVALID;

  auto it = n.begin();
  const auto& sub_n = it->second;
  if ( ! sub_n.IsMap()) return KPF::schema_style::INVALID;

  return KPF::validation_data::str_to_schema_style( it->first.as<string>() );
}

bool
cset_helper_parser( const string& context, const YAML::Node& n, KPFC::cset_t& cset )
{
  if ( ! n.IsMap() )
  {
    LOG_ERROR( main_logger, "YAML parser: " << context << ": not a map?" );
    return false;
  }
  try
  {
    for (auto i=n.begin(); i != n.end(); ++i)
    {
      cset.d.insert( make_pair( i->first.as<string>(), i->second.as<double>() ));
    }
  }
  catch (const std::invalid_argument& e )
  {
    LOG_ERROR( main_logger, "YAML parser: " << context << ": error converting to double: " << e.what() );
    return false;
  }
  return true;
}

bool
parse_poly( KPF::packet_t& p, const YAML::Node& n )
{
  if (! n.IsSequence())
  {
    LOG_ERROR( main_logger, "YAML parser: poly: not a sequence?" );
    LOG_ERROR( main_logger, "YAML parser: poly: as string: " << n.as<string>() );
    return false;
  }

  vector< pair < double, double > > xy;
  for (auto pt=n.begin(); pt != n.end(); ++pt)
  {
    const YAML::Node& npt( *pt );
    if ( ! npt.IsSequence() )
    {
      LOG_ERROR( main_logger, "YAML parser: poly: pt not a sequence?" );
      LOG_ERROR( main_logger, "YAML parser: poly: pt as string: " << npt.as<string>() );
      return false;
    }
    if ( npt.size() != 2)
    {
      LOG_ERROR( main_logger, "YAML parser: poly: pt has " << npt.size() << " points; expected 2" );
      return false;
    }
    try
    {
      xy.push_back( make_pair( stod(npt[0].as<string>()), stod( npt[1].as<string>() )));
    }
    catch (const std::invalid_argument& e)
    {
      LOG_ERROR( main_logger, "YAML parser: geometry: error converting to double: "
                 << e.what() );
      return false;
    }
  } // ... for all points

  new (&p.poly) KPFC::poly_t( xy );
  return true;
}

bool
parse_cset( KPF::packet_t& p, const YAML::Node& n )
{
  p.cset = new KPFC::cset_t();
  return cset_helper_parser( "cset", n, *(p.cset) );
}

bool
parse_geom( KPF::packet_t& p, const string& s )
{
  vector< string > tokens;
  ::kwiver::vital::tokenize( s, tokens, " ", kwiver::vital::TokenizeTrimEmpty );
  if ( tokens.size() != 4 )
  {
    LOG_ERROR( main_logger, "YAML parser: geometry: Expected four tokens from '"
               << s << "'; got " << tokens.size());
    return false;
  }
  double xy[4];
  try
  {
    for (auto i=0; i<4; ++i)
    {
      xy[i] = stod( tokens[ i ] );
    }
  }
  catch (const std::invalid_argument& e)
  {
    LOG_ERROR( main_logger, "YAML parser: geometry: error converting to double " << e.what() );
    for (auto i=0; i<4; ++i)
    {
      LOG_ERROR( main_logger, "index " << i << ": '" << tokens[ i ] << "'" );
    }
    return false;
  }

  new (&p.bbox) KPFC::bbox_t( xy[0], xy[1], xy[2], xy[3] );
  return true;
}

bool
parse_packet( const YAML::const_iterator& it, KPF::packet_t& p )
{
  bool okay = true;
  try
  {
    switch (p.header.style)
    {
    case KPF::packet_style::TS:
      new (&p.timestamp) KPFC::timestamp_t( it->second.as<double>() );
      break;

    case KPF::packet_style::CONF:
      new (&p.conf) KPFC::conf_t( it->second.as<double>() );
      break;

    case KPF::packet_style::CSET:
      okay = parse_cset( p, it->second );
      break;

    case KPF::packet_style::EVAL:
      new (&p.eval) KPFC::eval_t( it->second.as<double>() );
      break;

    case KPF::packet_style::ID:
      new (&p.id) KPFC::id_t( it->second.as<unsigned long long>() );
      break;

    case KPF::packet_style::KV:
      new (&p.kv) KPFC::kv_t( it->first.as<string>(), it->second.as<string>() );
      break;

    case KPF::packet_style::META:
      {
        string metadata_line = "";
        // first try it as a simple string
        try
        {
          metadata_line = it->second.as<string>();
        }
        catch (const YAML::Exception& e )
        {
          // hmm, it may have embedded yaml, try that
          YAML::Emitter meta_rewrite;
          meta_rewrite << it->second;
          metadata_line = string( meta_rewrite.c_str() );
        }
        new (&p.meta) KPFC::meta_t( metadata_line );
        p.header.domain = KPF::packet_header_t::NO_DOMAIN;
      }
      break;

    case KPF::packet_style::POLY:
      okay = parse_poly( p, it->second );
      break;

    case KPF::packet_style::GEOM:
      okay = parse_geom( p, it->second.as<string>() );
      break;

    default:
      {
        LOG_ERROR( main_logger, "No implementation for parsing packet of type " << p.header );
        okay = false;
      }
    } // ... switch
  } // ... try
  catch (const YAML::Exception& e )
  {
    LOG_ERROR( main_logger, "YAML exception parsing packet (header " << p.header << "): " << e.what() );
    okay = false;
  }

  return okay;
}

bool
parse_general( const YAML::Node& n, KPF::packet_buffer_t& local_packet_buffer )
{
  bool okay = true;
  for (auto it=n.begin(); (okay) && (it != n.end()); ++it)
  {
    auto h = KPF::packet_header_parser( it->first.as<string>() );

    KPF::packet_t p(h);
    okay = parse_packet( it, p );
    {
      ostringstream oss;
      if (okay) oss << p; else oss << "invalid";
    }
    if (okay)
    {
      local_packet_buffer.insert( make_pair( p.header, p ));
    }
    else
    {
      LOG_ERROR( main_logger, "parse general: couldn't parse packet of type " << h );
    }

  } // ...for all entries in this node's map (or until there's an error)
  return okay;
}


bool
parse_scoped_timespan( const YAML::Node& n,
                       vector< KPFC::scoped< KPFC::timestamp_range_t > >& tsvector )
{
  //
  // A scoped timepan appears in the YAML like this:
  //
  // timespan: [{ tsr0: [1001 , 1010],  }]
  //
  // which parses out like this:
  //
  // timespan => (sequence)
  //    (map)
  //      tsr0 => (sequence)
  //        (scalar) '1001'
  //        (scalar) '1010'
  //

  // iterate over each top-level sequence entry "A"
  for (auto seq = n.begin(); seq != n.end(); ++seq )
  {
    // each "A" is a map of tsrN -> sequence of two scalars
    for (auto i = seq->begin(); i != seq->end(); ++i)
    {
      auto h=KPF::packet_header_parser( i->first.as<string>() );
      if ( h.style != KPF::packet_style::TSR)
      {
        LOG_ERROR( main_logger, "timespan parser: couldn't parse timespan from '"
                   << i->first.as<string>() << "'" );
        return false;
      }
      const YAML::Node& tsr_payload = i->second;
      if (tsr_payload.size() != 2)
      {
        LOG_ERROR( main_logger, "timespan parser: sequence length " << tsr_payload.size()
                   << "; expected 2" );
        return false;
      }
      tsvector.push_back(
        KPFC::scoped< KPFC::timestamp_range_t>(
          KPFC::timestamp_range_t( tsr_payload[0].as<double>(), tsr_payload[1].as<double>() ),
          h.domain ));
    }
  }
  return true;
}

bool
parse_actors( const YAML::Node& n,
              vector< KPFC::activity_t::actor_t>& actors )
{
  //
  // Actors appear in the YAML like this:
  //
  //  actors: [{id1: 15, timespan: [{tsr0: [857, 897]}]} , {id1: 1, timespan: [{tsr0: [857, 897]}]} ,  ]
  //
  // which parses out like this:
  //
  //    actors => (sequence)
  //      (map)
  //        id1 => (scalar) '15'
  //        timespan => (sequence)
  //          (map)
  //            tsr0 => (sequence)
  //              (scalar) '857'
  //              (scalar) '897'
  //      (map)
  //        id1 => (scalar) '1'
  //        timespan => (sequence)
  //          (map)
  //            tsr0 => (sequence)
  //              (scalar) '857'
  //              (scalar) '897'

  for (auto seq=n.begin(); seq != n.end(); ++seq)
  {
    KPFC::activity_t::actor_t actor;
    for (auto actor_map = seq->begin(); actor_map != seq->end(); ++actor_map)
    {
      const string& s = actor_map->first.as<string>();
      if (s == "timespan" )
      {
        if (! parse_scoped_timespan( actor_map->second, actor.actor_timespan)) return false;
      }
      else
      {
        auto h = KPF::packet_header_parser( s );
        if ( h.style == KPF::packet_style::ID)
        {
          actor.actor_id = KPFC::scoped< KPFC::id_t >(
            KPFC::id_t( actor_map->second.as<unsigned long long>() ),
            h.domain );
        }
        else
        {
          LOG_WARN( main_logger, "Ignoring unexpected actor packet " << h );
        }
      }
    } // ...for nodes within the actor map

    actors.push_back( actor );

  } // ...for actors in the sequence

  return true;

}

bool
parse_activity( const YAML::Node& n, KPF::packet_buffer_t& local_packet_buffer )
{
  KPF::packet_t p;
  new (&p.activity) KPFC::activity_t();
  KPFC::activity_t& act = p.activity;

  // this is tricky-- if we see several ID packets, only take the one
  // with the domain matching the ACT tag. (What to do with the other ones?)
  // So we have to find the ACT tag before we can find the ID tag.

  act.activity_id = KPFC::scoped< KPFC::id_t >( KPFC::id_t(-1), -1 );

  // first we have to find the activity domain
  for (auto i=n.begin(); i != n.end(); ++i)
  {
    auto h = KPF::packet_header_parser( i->first.as<string>() );
    if ( h.style == KPF::packet_style::ACT)
    {
      act.activity_id.domain = h.domain;
      if (! cset_helper_parser( "activity-label cset", i->second, act.activity_labels ))
      {
        return false;
      }
    }
  }
  if ( act.activity_id.domain == -1 )
  {
    LOG_ERROR( main_logger, "No ACT tag in activity?" );
    return false;
  }

  // now we can loop through and deal with other packets
  for (auto i=n.begin(); i != n.end(); ++i)
  {
    // what is it?
    string s = i->first.as<string>();

    // special case the timespan and actors

    if ( s == "timespan" )
    {
      if (! parse_scoped_timespan( i->second, act.timespan )) return false;
    }
    else if (s == "actors" )
    {
      if (! parse_actors( i->second, act.actors )) return false;
    }

    // otherwise, we should at least be able to get a packet out of it
    else
    {
      auto h = KPF::packet_header_parser( s );

      if (( h.style == KPF::packet_style::ID) && ( h.domain == act.activity_id.domain ))
      {
        act.activity_id.t.d = i->second.as<unsigned long long>();
      }
      else if (h.style == KPF::packet_style::KV)
      {
        act.attributes.push_back( KPFC::kv_t( i->first.as<string>(),
                                              i->second.as<string>() ));
      }
      else if (h.style == KPF::packet_style::EVAL)
      {
        KPF::packet_t p(h);
        if ( ! parse_packet( i, p )) return false;
        act.evals.push_back ( {p.eval, p.header.domain} );
      }
      else if (h.style == KPF::packet_style::ACT)
      {
        // already handled above; ignore here to suppress warning message
      }
      else
      {
        LOG_WARN( main_logger, "Activity ignoring packet " << h << "?" );
      }
    }
  } // ...for all nodes

  //
  // Create the header and insert into the packet buffer
  //

  p.header = KPF::packet_header_t( KPF::packet_style::ACT, act.activity_id.domain );
  local_packet_buffer.insert( make_pair( p.header, p ));
  return true;
}

bool
parse_as_kpf_v3( const YAML::Node& n,
                 KPF::schema_style schema,
                 KPF::packet_buffer_t& local_packet_buffer )
{
  if (schema == KPF::schema_style::ACT)
  {
    return parse_activity( n, local_packet_buffer );
  }

  //
  // nothing else really requires special processing (yet)
  //

  return parse_general( n, local_packet_buffer );
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
  : current_record_schema( schema_style::INVALID )
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

bool
kpf_yaml_parser_t
::eof() const
{
  return this->current_record == this->root.end();
}

schema_style
kpf_yaml_parser_t
::get_current_record_schema() const
{
  return this->current_record_schema;
}

/**
 * \brief Read the next child of root into the packet buffer.
 *
 * Starting with KPFv3, each yaml line is either:
 *
 * meta: xxxxxx
 *
 * or a set of
 *
 * schema: {xxx}
 *
 * where "schema" is a tag meant to assist in validation and parsing.
 *
 * The result is still to populate the local packet buffer; the schema
 * does not propagate beyond the parser.
 *
 * There is still only one schema instance per line; we don't support
 * (for example)
 *
 * - { geom: { geom-stuff }, geom: {geom-stuff} }
 *
 */

bool
kpf_yaml_parser_t
::parse_next_record( packet_buffer_t& local_packet_buffer )
{

  //
  // This routine gets called once per line.
  //

  if (this->current_record == this->root.end())
  {
    return false;
  }

  const YAML::Node& n = *(this->current_record++);

  //
  // The line must be a map.
  //

  if (! n.IsMap())
  {
    LOG_ERROR( main_logger, "YAML: root node is " << n.Type() << "; expected map" );
    LOG_ERROR( main_logger, "node as string: '" << n.as<string>() << "'" );
    return false;
  }

  //
  // Each element of the map is itself a map, k:v. If k is a kpfv3 schema,
  // parse v as a map of elements which must conform to the constraints of
  // the schema k. Otherwise, treat all of the toplevel map as a kpfv2 file,
  // and guess at its schema.
  //

  //
  // Is it a kpfv3 schema tag?
  //

  bool okay = false;
  auto check = kpf_v3_check( n );
  auto n_sub = n.begin();

  this->current_record_schema = check;

  // special case for meta
  if ( str2style( n_sub->first.as<string>()) == packet_style::META )
  {
    okay = parse_as_kpf_v3( n, schema_style::UNSPECIFIED, local_packet_buffer );
  }
  else
  {
    if ( check != schema_style::INVALID )
    {
      okay = parse_as_kpf_v3( n_sub->second, check, local_packet_buffer );
    }
  }

  return okay;
}

} // ...kpf
} // ...vital
} // ...kwiver
