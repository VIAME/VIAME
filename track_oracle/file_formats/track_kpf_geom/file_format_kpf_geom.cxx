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


/**
 * @file
 * @brief The track_oracle file format implementation for KPF geometry.
 *
 */

#include "file_format_kpf_geom.h"

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

#include <track_oracle/data_terms/data_terms.h>
#include <arrows/kpf/yaml/kpf_reader.h>
#include <arrows/kpf/yaml/kpf_yaml_parser.h>
#include <vital/util/tokenize.h>
#include <track_oracle/utils/logging_map.h>

#include <yaml-cpp/yaml.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

using std::string;
using std::istream;
using std::ifstream;
using std::stringstream;
using std::ostringstream;
using std::map;

namespace KPF=::kwiver::vital::kpf;

namespace // anon
{
using namespace kwiver::track_oracle;



struct kpf_geom_exception
{
  explicit kpf_geom_exception( const string& msg ): what(msg) {}
  string what;
};

void
add_to_row( kwiver::logging_map_type& log_map,
            const oracle_entry_handle_type& row,
            const KPF::packet_t& p )
{
  namespace KPFC = ::kwiver::vital::kpf::canonical;

  //
  // Step through the packet types and add them
  //
  switch (p.header.style)
  {
  case KPF::packet_style::ID:
    {
      unsigned id = p.id.d;
      switch (p.header.domain)
      {
      case KPFC::id_t::DETECTION_ID:
        {
          track_field< dt::detection::detection_id > x;
          x(row) = id;
        }
        break;
      case KPFC::id_t::TRACK_ID:
        {
          track_field< dt::tracking::external_id > x;
          x(row) = id;
        }
        break;
      case KPFC::id_t::EVENT_ID:
        {
          track_field< dt::events::event_id > x;
          x(row) = id;
        }
      default:
        {
          ostringstream oss;
          oss << "Ignoring ID domain " << p.header.domain;
          log_map.add_msg( oss.str() );
        }
      }
    }
    break;

  case KPF::packet_style::TS:
    {
      switch (p.header.domain)
      {
      case KPFC::timestamp_t::FRAME_NUMBER:
        {
          track_field< dt::tracking::frame_number > x;
          x(row) = static_cast< dt::tracking::frame_number::Type >( p.timestamp.d );
        }
        break;
      default:
        {
          ostringstream oss;
          oss << "Ignoring TS domain " << p.header.domain;
          log_map.add_msg( oss.str() );
        }
      }
    }
    break;
  case KPF::packet_style::GEOM:
    {
      switch (p.header.domain)
      {
      case KPFC::bbox_t::IMAGE_COORDS:
        {
          track_field< dt::tracking::bounding_box > x;
          x(row) = vgl_box_2d<double>(
            vgl_point_2d<double>( p.bbox.x1, p.bbox.y1 ),
            vgl_point_2d<double>( p.bbox.x2, p.bbox.y2 ));
        }
        break;
      default:
        {
          ostringstream oss;
          oss << "Ignoring TS domain " << p.header.domain;
          log_map.add_msg( oss.str() );
        }
      }
    }
    break;
  case KPF::packet_style::CONF:
    {
      // since KPF's domains for confidence are open-ended, we don't
      // hardwire this file's confidence values to fixed data_terms
      // the way we do with (for example) the IDs or bounding boxes--
      // that's the whole rationale for pre-defined domains, that we
      // can assert their semantic interpretation. Not so with conf packets,
      // so we'll just create their storage on-the-fly.

      ostringstream oss;
      oss << KPF::style2str( p.header.style ) << "_" << p.header.domain;
      field_handle_type f = track_oracle_core::lookup_by_name( oss.str() );
      if ( f == INVALID_FIELD_HANDLE )
      {
        element_descriptor e( oss.str(),
                              "KPF ad-hoc",
                              typeid( double(0) ).name(),
                              element_descriptor::ADHOC );
        f = track_oracle_core::create_element< double >( e );
      }
      track_oracle_core::get_field<double>( row, f ) = p.conf.d;
    }
    break;
  case KPF::packet_style::KV:
    {
      // Like the conf values, but here we don't have a domain, and
      // the type is string, not double.

      ostringstream oss;
      oss << KPF::style2str( p.header.style ) << "_" << p.kv.key;
      field_handle_type f = track_oracle_core::lookup_by_name( oss.str() );
      if ( f == INVALID_FIELD_HANDLE )
      {
        element_descriptor e( oss.str(),
                              "KPF ad-hoc",
                              typeid( string("") ).name(),
                              element_descriptor::ADHOC );
        f = track_oracle_core::create_element< string >( e );
      }
      track_oracle_core::get_field<string>( row, f ) = p.kv.val;
    }
    break;
  default:
    {
      ostringstream oss;
      oss << "Ignoring packet header " << p.header;
      log_map.add_msg( oss.str() );
    }
  };
}

} // ...anon

namespace kwiver {
namespace track_oracle {

file_format_kpf_geom
::file_format_kpf_geom()
  : file_format_base( TF_KPF_GEOM, "KPF geometry" )
{
  this->globs.push_back("*.geom.yml");

}

file_format_kpf_geom
::~file_format_kpf_geom()
{
}

bool
file_format_kpf_geom
::inspect_file( const string& fn ) const
{
  // This just checks that it's YAML, not that it's a geom schema
  // (would have to trawl through metadata packets line-by-line)
  ifstream is( fn.c_str() );
  if ( ! is ) return false;

  string s;
  if ( ! std::getline( is, s )) return false;

  stringstream ss(s);

  bool parsed = true;
  try
  {
    YAML::Load( ss );
  }
  // Can't catch the exact YAML exception on OSX
  // see https://stackoverflow.com/questions/21737201/problems-throwing-and-catching-exceptions-on-os-x-with-fno-rtti
  catch ( ... )
  {
    parsed = false;
  }
  return parsed;
}

bool
file_format_kpf_geom
::read( const string& fn, track_handle_list_type& tracks ) const
{
  ifstream is( fn.c_str() );
  return is && this->read( is, tracks );
}

bool
file_format_kpf_geom
::read( istream& is, track_handle_list_type& tracks ) const
{
  LOG_INFO( main_logger, "KPF geometry YAML load start");
  KPF::kpf_yaml_parser_t parser( is );
  KPF::kpf_reader_t reader( parser );
  LOG_INFO( main_logger, "KPF geometry YAML load end");
  track_kpf_geom_type entry;

  size_t rc = 0;
  map< unsigned, track_handle_type > track_map;
  logging_map_type wmsgs( main_logger, KWIVER_LOGGER_SITE );

  while ( reader.next() )
  {
    //
    // Where to store meta packets? Are they associated with the source
    // file? Particular tracks? TBD!
#if 0
    for (auto m: reader.get_meta_packets() )
    {
    }
#endif

    ++rc;
    try
    {
      namespace KPFC = ::kwiver::vital::kpf::canonical;

      auto track_id_probe = reader.transfer_packet_from_buffer(
        KPF::packet_header_t( KPF::packet_style::ID, KPFC::id_t::TRACK_ID ));
      if ( ! track_id_probe.first )
      {
        ostringstream oss;
        oss << "Missing packet ID1 on non-meta record line " << rc << "; skipping";
        throw kpf_geom_exception( oss.str() );
      }

      auto track_probe = track_map.find( track_id_probe.second.id.d );
      if (track_probe == track_map.end())
      {
        track_map[ track_id_probe.second.id.d ] = entry.create();
      }
      track_handle_type t = track_map[ track_id_probe.second.id.d ];

      // the track ID is added to the track row;
      // other packets are added to the frames
      add_to_row( wmsgs, t.row, track_id_probe.second );

      frame_handle_type f = entry(t).create_frame();

      struct kpf_loader
      {
        KPF::packet_header_t h;
        bool required;
      };

      kpf_loader fields[] = {
        { KPF::packet_header_t( KPF::packet_style::GEOM, KPFC::bbox_t::IMAGE_COORDS ), true },
        { KPF::packet_header_t( KPF::packet_style::TS, KPFC::timestamp_t::FRAME_NUMBER ), true },
        { KPF::packet_header_t( KPF::packet_style::ID, KPFC::id_t::DETECTION_ID ), true }
      };

      for (const auto& field: fields)
      {
        auto probe = reader.transfer_packet_from_buffer( field.h );
        if ( field.required && (! probe.first ))
        {
          ostringstream oss;
          oss << "Missing packet " << KPF::style2str( field.h.style ) << ":"
              << field.h.domain << " in non-meta record " << rc;
          throw kpf_geom_exception( oss.str() );
        }
        add_to_row( wmsgs, f.row, probe.second );
      }

      //
      // If any other packets are present (confidences, kv) then store them
      // as well
      //

      for (const auto& p: reader.get_packet_buffer())
      {
        add_to_row( wmsgs, f.row, p.second );
      }

    }
    catch ( const kpf_geom_exception& e )
    {
      LOG_ERROR( main_logger, "Error parsing KPF geom: " << e.what << "; skipping" );

      // hmm, t and/or f "leak"; we never add them to the returned track set,
      // but still, it's messy
    }

    reader.flush();
  }
  for (auto p: track_map)
  {
    tracks.push_back( p.second );
  }
  if (! wmsgs.empty() )
  {
    LOG_INFO( main_logger, "KPF geom parsing warnings begin" );
    wmsgs.dump_msgs();
    LOG_INFO( main_logger, "KPF geom parsing warnings end" );
  }
  return true;
}

bool
file_format_kpf_geom
::write( const std::string& fn,
         const track_handle_list_type& tracks) const
{
  return false;
}

bool
file_format_kpf_geom
::write( std::ostream& os,
         const track_handle_list_type& tracks) const
{
  return false;
}

} // ...track_oracle
} // ...kwiver
