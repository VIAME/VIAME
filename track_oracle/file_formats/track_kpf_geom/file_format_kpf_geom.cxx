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
#include <arrows/kpf/yaml/kpf_yaml_writer.h>
#include <arrows/kpf/yaml/kpf_yaml_parser.h>
#include <arrows/kpf/yaml/kpf_canonical_io_adapter.h>
#include <vital/util/tokenize.h>
#include <track_oracle/utils/logging_map.h>

#include <kwiversys/RegularExpression.hxx>

#include <yaml-cpp/yaml.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <cctype>

using std::string;
using std::istream;
using std::ifstream;
using std::ofstream;
using std::stringstream;
using std::ostringstream;
using std::map;
using std::stoi;

namespace KPF=::kwiver::vital::kpf;

namespace // anon
{
using namespace kwiver::track_oracle;

struct vgl_box_adapter_t: public KPF::kpf_box_adapter< vgl_box_2d<double> >
{
  vgl_box_adapter_t():
    kpf_box_adapter< vgl_box_2d<double> >(
      // reads the canonical box "b" into the vgl_box "d"
      []( const KPF::canonical::bbox_t& b, vgl_box_2d<double>& d ) {
        d = vgl_box_2d<double>(
          vgl_point_2d<double>( b.x1, b.y1 ),
          vgl_point_2d<double>( b.x2, b.y2 ));
      },

      // converts a vgl box "d" into a canonical box and returns it
      []( const vgl_box_2d<double>& d ) {
        return KPF::canonical::bbox_t(
          d.min_x(),
          d.min_y(),
          d.max_x(),
          d.max_y()); })
  {}
};


struct kpf_geom_exception
{
  explicit kpf_geom_exception( const string& msg ): what(msg) {}
  string what;
};

//
// When looking in track_oracle's database for any optional fields
// to emit, we select anything whose field name matches either
//
// XXX_nnn, where 'XXX' can be converted to a KPF style, and nnn is an
// integer (interpreted as the domain), and the type is a double,
//
// OR
//
// key_KKK, where 'key' is the KPF style for a key/value pair; KKK is
// taken as the key, and the track_oracle entry is the value, and the type
// is a string.
//
// Another approach might be to store these packet headers when we read the
// KPF in, but somehow doing it just-in-time here feels better, if more
// convoluted.
//

//
// Whoops, we have to store a whole packet because the KV packet
// doesn't record the key in the header. Drat.
//

map< field_handle_type, KPF::packet_t >
get_optional_fields()
{
  map< field_handle_type, KPF::packet_t > optional_fields;
  string dbltype( typeid( double(0) ).name()), strtype( typeid( string("") ).name());
  string kpf_key_str( KPF::style2str( KPF::packet_style::KV ));
  // Note that these regexs are tied to the synthesized field names in the reader.
  kwiversys::RegularExpression field_dbl("^([a-zA-Z0-9]+)_([0-9]+)$");
  kwiversys::RegularExpression field_kv("^"+kpf_key_str+"_([a-zA-Z0-9]+)$");

  for (auto i: track_oracle_core::get_all_field_handles())
  {
    element_descriptor e = track_oracle_core::get_element_descriptor( i );

    // Is the type double?
    if (e.typeid_str == dbltype)
    {
      // did we find two matches and is the first a KPF style?
      if (field_dbl.find( e.name ))
      {
        auto style = KPF::str2style( field_dbl.match(1) );
        if (style != KPF::packet_style::INVALID )
        {
          // Convert the domain and associate a packet with the field
          optional_fields[i] = KPF::packet_t( KPF::packet_header_t( style, stoi( field_dbl.match(2) )));

        } // ...not a valid KPF domain
      } // ... didn't find two matches
    } // ...field is not a double

    // Is the type string?
    else if (e.typeid_str == strtype)
    {
      if (field_kv.find( e.name ))
      {
        // Create a partial KV packet with the key
        KPF::packet_t p( (KPF::packet_header_t( KPF::packet_style::KV )) );
        p.kv.key = field_kv.match(1);
        optional_fields[i] = p;

      } // ...didn't get exactly one match
    } // ... field is not a string
  } // ...for all fields in track_oracle

  return optional_fields;
}

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


      // Note that this string needs to be kept in sync with the regex
      // used to pick it up in the writer.

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

      // Note that this string needs to be kept in sync with the regex
      // used to pick it up in the writer.

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
  //
  // Load the YAML
  //

  LOG_INFO( main_logger, "KPF geometry YAML load start");
  KPF::kpf_yaml_parser_t parser( is );
  KPF::kpf_reader_t reader( parser );
  LOG_INFO( main_logger, "KPF geometry YAML load end");
  track_kpf_geom_type entry;

  size_t rc = 0;
  map< unsigned, track_handle_type > track_map;
  logging_map_type wmsgs( main_logger, KWIVER_LOGGER_SITE );

  //
  // process each line
  //

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

      //
      // special handling for the track ID (aka ID1) since we need
      // that as our key into track_oracle's track/frame pool
      //

      auto track_id_probe = reader.transfer_packet_from_buffer(
        KPF::packet_header_t( KPF::packet_style::ID, KPFC::id_t::TRACK_ID ));
      if ( ! track_id_probe.first )
      {
        ostringstream oss;
        oss << "Missing packet ID1 on non-meta record line " << rc << "; skipping";
        throw kpf_geom_exception( oss.str() );
      }

      //
      // do we already have a track_oracle track structure for this ID?
      // if not, create one
      //

      auto track_probe = track_map.find( track_id_probe.second.id.d );
      if (track_probe == track_map.end())
      {
        track_map[ track_id_probe.second.id.d ] = entry.create();
      }
      track_handle_type t = track_map[ track_id_probe.second.id.d ];

      // the track ID is added to the track row;
      // other packets are added to the frames
      add_to_row( wmsgs, t.row, track_id_probe.second );

      //
      // create a frame for this record and read the rest of the packets into it
      //

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

  //
  // convert the map of track handles into the return buffer
  //

  for (auto p: track_map)
  {
    tracks.push_back( p.second );
  }

  //
  // all done; emit any warnings and return
  //

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
  ofstream os( fn.c_str() );
  if ( ! os )
  {
    LOG_ERROR( main_logger, "Couldn't open '" << fn << "' for writing" );
    return false;
  }
  return this->write( os, tracks );
}

bool
file_format_kpf_geom
::write( std::ostream& os,
         const track_handle_list_type& tracks) const
{
  namespace KPFC = KPF::canonical;
  logging_map_type wmsgs( main_logger, KWIVER_LOGGER_SITE );
  KPF::record_yaml_writer w( os );
  track_kpf_geom_type entry;
  vgl_box_adapter_t box_adapter;

  auto optional_fields = get_optional_fields();
  for (auto i: optional_fields)
  {
    const KPF::packet_t& p = i.second;
    if (p.header.style == KPF::packet_style::KV)
    {
      LOG_INFO( main_logger, "KPF geom writer: adding optional KV " << p.kv.key );
    }
    else
    {
      LOG_INFO( main_logger, "KPF geom writer: adding optional " << p.header );
    }
  }

  //
  // worry about meta packets later
  //

  for (const auto& t: tracks )
  {
    for (const auto& f: track_oracle_core::get_frames( t ) )
    {
      //
      // write out the required fields
      //

      w
        << KPF::writer< KPFC::id_t >( entry(t).track_id(), KPFC::id_t::TRACK_ID )
        << KPF::writer< KPFC::id_t >( entry[f].det_id(), KPFC::id_t::DETECTION_ID )
        << KPF::writer< KPFC::bbox_t >( box_adapter( entry[f].bounding_box()), KPFC::bbox_t::IMAGE_COORDS )
        << KPF::writer< KPFC::timestamp_t >( entry[f].frame_number(), KPFC::timestamp_t::FRAME_NUMBER );

      //
      // write out the optional fields
      //

      for (auto fh: track_oracle_core::fields_at_row( f.row ) )
      {
        auto probe = optional_fields.find( fh );
        if (probe == optional_fields.end())
        {
          continue;
        }

        //
        // hmm, awkward
        //

        const KPF::packet_t& p = probe->second;
        switch (p.header.style)
        {
        case KPF::packet_style::CONF:
          {
            auto v = track_oracle_core::get<double>( f.row, fh );
            if ( v.first )
            {
              w << KPF::writer< KPFC::conf_t >( v.second, p.header.domain );
            }
            else
            {
              LOG_ERROR( main_logger, "Lost value for " << p.header << "?" );
            }
          }
          break;
        case KPF::packet_style::EVAL:
          {
            auto v = track_oracle_core::get<double>( f.row, fh );
            if ( v.first )
            {
              w << KPF::writer< KPFC::eval_t >( v.second, p.header.domain );
            }
            else
            {
              LOG_ERROR( main_logger, "Lost value for " << p.header << "?" );
            }
          }
          break;
        case KPF::packet_style::KV:
          {
            auto v = track_oracle_core::get<string>( f.row, fh );
            if ( v.first )
            {
              w << KPF::writer< KPFC::kv_t >( p.kv.key, v.second );
            }
            else
            {
              LOG_ERROR( main_logger, "Lost value for " << p.header << "?" );
            }
          }
          break;
        default:
          {
            ostringstream oss;
            oss << "No handler for optional packet " << p.header;
            wmsgs.add_msg( oss.str() );
          }
        }
      } // ...for all optional fields

      w << KPF::record_yaml_writer::endl;

    } // ... for all frames
  } // ... for all tracks

  if (! wmsgs.empty() )
  {
    LOG_INFO( main_logger, "KPF geom writer warnings begin" );
    wmsgs.dump_msgs();
    LOG_INFO( main_logger, "KPF geom writer warnings end" );
  }

  return true;
}

} // ...track_oracle
} // ...kwiver
