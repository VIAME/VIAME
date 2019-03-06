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

/**
 * @file
 * @brief Support for predefined KPF yaml schemas.
 *
 */

#include "kpf_yaml_schemas.h"

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( "arrows.kpf.kpf_yaml_schemas" ) );

using std::vector;
using std::string;

namespace // anon
{
namespace KPF = kwiver::vital::kpf;

vector< KPF::validation_data >
required_packets( KPF::schema_style schema )
{
  switch (schema )
  {
  case KPF::schema_style::GEOM:
    return {
      KPF::validation_data( KPF::packet_style::ID ),
      KPF::validation_data( KPF::packet_style::TS ),
      KPF::validation_data( KPF::packet_style::GEOM )
    };

  case KPF::schema_style::ACT:
    return {
      KPF::validation_data( KPF::packet_style::ACT ),
      KPF::validation_data( KPF::packet_style::ID ),
      KPF::validation_data( KPF::packet_style::KV, "timespan" ),
      KPF::validation_data( KPF::packet_style::KV, "actors" )
    };

  case KPF::schema_style::TYPES:
    return {
      KPF::validation_data( KPF::packet_style::ID ),
      KPF::validation_data( KPF::packet_style::CSET )
    };

  case KPF::schema_style::REGIONS:
    return {
      KPF::validation_data( KPF::packet_style::ID ),
      KPF::validation_data( KPF::packet_style::TS ),
      KPF::validation_data( KPF::packet_style::POLY )
    };

  default:
    LOG_ERROR( main_logger, "Unhandled validation schema request for "
               << static_cast<int>( schema ) );
    // this should never be present and thus should trigger an
    // invalidation upstream
    return {
      KPF::validation_data( KPF::packet_style::INVALID )
    };
  }
}

} // ...anon


namespace kwiver {
namespace vital {
namespace kpf {

string
validation_data
::schema_style_to_str( schema_style s )
{
  switch (s)
  {
  case schema_style::INVALID:     return "invalid";
  case schema_style::META:        return "meta";
  case schema_style::GEOM:        return "geom";
  case schema_style::ACT:         return "act";
  case schema_style::TYPES:       return "types";
  case schema_style::REGIONS:     return "regions";
  case schema_style::UNSPECIFIED: return "unspecified";
  default: return "invalid";
  }
}

schema_style
validation_data
::str_to_schema_style( const string& s )
{
  for (auto style: { schema_style::INVALID, schema_style::META, schema_style::GEOM, schema_style::ACT,
        schema_style::TYPES, schema_style::REGIONS, schema_style::UNSPECIFIED } )
  {
    if (s == schema_style_to_str( style ))
    {
      return style;
    }
  }
  return schema_style::INVALID;
}

validation_data
::validation_data( YAML::const_iterator it )
{
  // This is annoying; at the top-level, we'll see:
  // act17 => (scalar) 'walking'
  // id17 => (scalar) '100'
  // myconf => (scalar) '0.3'
  // timespan => (sequence)
  //   (map)
  //     tsr0 => (sequence)
  //     (etc)
  //
  // i.e. 'act17' and 'id17' should both come back as act and
  // id packet headers, respectively, but 'myconf' and 'timespan'
  // will default to KV. However, timespan is *NOT* a KV.
  // Represent this as a validation_data packet with a null key.

  auto h = packet_header_parser( it->first.as<string>() );
  LOG_DEBUG( main_logger, "vd ctor " << h );
  this->style = h.style;
  this->key = "";
  if ( this->style == packet_style::KV )
  {
      this->key = it->first.as<string>();
  }
}

vector< validation_data >
validate_schema( schema_style schema, const packet_buffer_t& packets )
{
  //
  // Check if the required packets are present; return a list of those
  // that aren't; i.e. empty means validated
  //

  vector< validation_data > ret;

  // quick-exit for the no-exam-required cases
  if ( schema == schema_style::INVALID ) return ret;
  if ( schema == schema_style::UNSPECIFIED) return ret;

  const auto& req = required_packets( schema );
  LOG_DEBUG( main_logger, "Looking for required '" << validation_data::schema_style_to_str( schema ));

  for (const auto& v: req )
  {
  LOG_DEBUG( main_logger, "verification check for " << style2str( v.style ) << " / '" << v.key << "'" );

    bool found = false;
    for (auto p = packets.begin(); (! found ) && p != packets.end(); ++p )
    {
      // look at each packet (quick exit if found); if the styles match
      // it's found UNLESS it's a key, in which case we actually have to
      // look into the packet
      if (v.style == p->second.header.style)
      {
        if ( v.style != KPF::packet_style::KV )
        {
          found = true;
        }
        else
        {
          if (v.key == p->second.kv.key)
          {
            found = true;
          }
        }
      }
      LOG_DEBUG( main_logger, "...vs " << p->second << "? " << found );
    } // ... for each packet

    LOG_DEBUG( main_logger, "Found: " << found );

    if ( ! found )
    {
      ret.push_back( v );
    }

  } // ... for all requirements

  return ret;
}

vector< validation_data >
validate_schema( schema_style schema, const vector< validation_data>& vpackets )
{
  //
  // Check if the required packets are present; return a list of those
  // that aren't; i.e. empty means validated
  //

  vector< validation_data > ret;

  // quick-exit for the no-exam-required cases
  if ( schema == schema_style::INVALID ) return ret;
  if ( schema == schema_style::UNSPECIFIED) return ret;

  const auto& req = required_packets( schema );
  LOG_DEBUG( main_logger, "Looking for required '" << validation_data::schema_style_to_str( schema ) << "'" );


  for (const auto& v: req )
  {
    LOG_DEBUG( main_logger, "verification check for " << style2str( v.style ) << " / '" << v.key << "'" );
    bool found = false;
    for (auto p = vpackets.begin(); (! found ) && p != vpackets.end(); ++p )
    {
      if (v.style == p->style)
      {
        if ( v.style == KPF::packet_style::KV )
        {
          found = (v.key == p->key);
        }
        else
        {
          found = true;
        }
      }
      LOG_DEBUG( main_logger, "...vs " << style2str(p->style) << "/ '" << p->key << "' ? " << found );
    } // ... for each packet

    LOG_DEBUG( main_logger, "Found: " << found );

    if ( ! found )
    {
      ret.push_back( v );
    }

  } // ... for all requirements

  return ret;
}


} // ...kpf
} // ...vital
} // ...kwiver
