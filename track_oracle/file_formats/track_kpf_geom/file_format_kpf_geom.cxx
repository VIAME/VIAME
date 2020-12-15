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
#include <track_oracle/file_formats/kpf_utils/kpf_utils.h>

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
using std::vector;
using std::stoi;

namespace KPF=::kwiver::vital::kpf;
namespace dt = ::kwiver::track_oracle::dt;

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
  map< dt::tracking::external_id::Type, track_handle_type > track_map;
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
      kpf_utils::add_to_row( wmsgs, t.row, track_id_probe.second );

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
        kpf_utils::add_to_row( wmsgs, f.row, probe.second );
      }

      //
      // If any other packets are present (confidences, kv) then store them
      // as well
      //

      for (const auto& p: reader.get_packet_buffer())
      {
        kpf_utils::add_to_row( wmsgs, f.row, p.second );
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

  //
  // worry about meta packets later
  //

  kpf_utils::optional_field_state ofs( wmsgs );

  for (const auto& t: tracks )
  {
    for (const auto& f: track_oracle_core::get_frames( t ) )
    {
      //
      // write out the required fields
      //

      w.set_schema( KPF::schema_style::GEOM)
        << KPF::writer< KPFC::id_t >( entry(t).track_id(), KPFC::id_t::TRACK_ID )
        << KPF::writer< KPFC::id_t >( entry[f].det_id(), KPFC::id_t::DETECTION_ID )
        << KPF::writer< KPFC::bbox_t >( box_adapter( entry[f].bounding_box()), KPFC::bbox_t::IMAGE_COORDS )
        << KPF::writer< KPFC::timestamp_t >( entry[f].frame_number(), KPFC::timestamp_t::FRAME_NUMBER );

      //
      // write out the optional fields
      //

      vector< KPF::packet_t > opt_packets =
        kpf_utils::optional_fields_to_packets( ofs, f.row );
      kpf_utils::write_optional_packets( opt_packets, wmsgs, w );

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
