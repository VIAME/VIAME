/*ckwg +5
 * Copyright 2012-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "file_format_kw18.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>

#include <vul/vul_file.h>
#include <vul/vul_timer.h>
#include <vul/vul_sprintf.h>

#include <vgl/vgl_box_2d.h>
#include <vgl/vgl_point_2d.h>
#include <vgl/vgl_point_3d.h>

#include <vital/util/string.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::getline;
using std::ifstream;
using std::istream;
using std::map;
using std::ofstream;
using std::ostream;
using std::ostringstream;
using std::pair;
using std::sscanf;
using std::streamsize;
using std::string;

namespace // anon
{

// utility function to read text files, skipping blank and '#'-comment lines
// return true and populate param if we can read a line, otherwise return false

bool
get_next_nonblank_line( istream& is, string& line )
{
  while ( getline(is, line) )
  {
    kwiver::vital::left_trim(line);
    // skip blank lines
    if (line.empty())
    {
      continue;
    }
    // skip comments
    if (line[0] == '#')
    {
      continue;
    }

    return true;
  }
  return false;
}

struct kw18_line_parser
{
  // matlab always writes doubles
  double dbl_track_id, dbl_n_frames, dbl_frame_num; // fields 0, 1, 2
  double loc_x, loc_y, vel_x, vel_y; // fields 3, 4, 5, 6
  double obj_loc_x, obj_loc_y;   // fields 7,8
  double bb_c1_x, bb_c1_y, bb_c2_x, bb_c2_y; // fields 9, 10, 11, 12
  double area, world_x, world_y, world_z; // fields 13, 14, 15, 16
  double timestamp; // field 17
  double kw19; // field 18; not always present

  bool read_19_columns;

  explicit kw18_line_parser( const kwiver::track_oracle::kw18_reader_opts& opts )
  {
    this->read_19_columns = opts.kw19_hack;
  }

  bool parse( const string& s )
  {
    if ( ! this->read_19_columns )
    {
      this->kw19 = 0.0;
      return (sscanf( s.c_str(),
                      "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                      &this->dbl_track_id,
                      &this->dbl_n_frames,
                      &this->dbl_frame_num,
                      &this->loc_x,
                      &this->loc_y,
                      &this->vel_x,
                      &this->vel_y,
                      &this->obj_loc_x,
                      &this->obj_loc_y,
                      &this->bb_c1_x,
                      &this->bb_c1_y,
                      &this->bb_c2_x,
                      &this->bb_c2_y,
                      &this->area,
                      &this->world_x,
                      &this->world_y,
                      &this->world_z,
                      &this->timestamp ) == 18 );
    }
    else
    {
      return (sscanf( s.c_str(),
                      "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                      &this->dbl_track_id,
                      &this->dbl_n_frames,
                      &this->dbl_frame_num,
                      &this->loc_x,
                      &this->loc_y,
                      &this->vel_x,
                      &this->vel_y,
                      &this->obj_loc_x,
                      &this->obj_loc_y,
                      &this->bb_c1_x,
                      &this->bb_c1_y,
                      &this->bb_c2_x,
                      &this->bb_c2_y,
                      &this->area,
                      &this->world_x,
                      &this->world_y,
                      &this->world_z,
                      &this->timestamp,
                      &this->kw19 ) == 19 );
    }
  }
};

template< typename T >
typename ::kwiver::track_oracle::track_field<T>::Type
logged_get_field( ::kwiver::track_oracle::track_field<T>& tf,
                  ::kwiver::track_oracle::oracle_entry_handle_type row,
                  map< string, size_t >& warnings )
{
  pair< bool, typename ::kwiver::track_oracle::track_field<T>::Type > probe = tf.get( row );
  if (! probe.first )
  {
    ++warnings[ tf.get_field_name() ];
  }
  return probe.second;
}

template< typename T >
string
logged_output( const ::kwiver::track_oracle::track_field< T >& tf,
               ::kwiver::track_oracle::oracle_entry_handle_type row,
               map< string, size_t >& warnings )
{
  // T is e.g. a data_term< vgl_box_2d<double> >.

  pair< bool, typename ::kwiver::track_oracle::track_field<T>::Type > probe = tf.get( row );
  // ... e.g. pair< bool, vgl_box_2d<double> >.
  if ( ! probe.first )
  {
    ++warnings[ tf.get_field_name() ];
  }

  ostringstream oss;
  oss  << tf.io_fmt( probe.second );
  return oss.str();
}


} // anon

namespace kwiver {
namespace track_oracle {

kw18_reader_opts&
kw18_reader_opts
::operator=( const file_format_reader_opts_base& rhs_base )
{
  const kw18_reader_opts* rhs = dynamic_cast< const kw18_reader_opts*>( &rhs_base );
  if (rhs)
  {
    this->set_kw19_hack( rhs->kw19_hack );
  }
  else
  {
    LOG_ERROR( main_logger, "Assigned non-kw18 options structure to kw18 opts; slicing" );
  }
  return *this;
}

bool
file_format_kw18
::inspect_file( const string& fn ) const
{
  ifstream is( fn.c_str() );
  if ( ! is )
  {
    LOG_ERROR( main_logger, "Couldn't open '" << fn << "'" );
    return false;
  }

  string line;
  // return true because the reader will also accept an empty file
  if ( ! get_next_nonblank_line( is, line )) return true;

  kw18_line_parser p( this->opts );
  return p.parse( line );
}

bool
file_format_kw18
::read( const string& fn,
        track_handle_list_type& tracks ) const
{
  ifstream is( fn.c_str() );
  if ( ! is )
  {
    LOG_ERROR( main_logger, "Couldn't open '" << fn << "'" );
    return false;
  }

  return this->internal_stream_read( is, vul_file::size(fn), tracks );
}


bool
file_format_kw18
::read( istream& is,
        track_handle_list_type& tracks ) const
{
  return this->internal_stream_read( is, 0, tracks );
}


bool
file_format_kw18
::internal_stream_read( istream& is,
                        size_t file_size,
                        track_handle_list_type& tracks ) const
{
  track_kw18_type kw18;
  string tmp;
  kw18_line_parser p( this->opts );
  track_field< double > relevancy( "relevancy" );

  bool current_external_id_valid = false;
  unsigned current_external_id = 0;

  vul_timer timer;
  size_t record=0;
  while( get_next_nonblank_line(is, tmp) )
  {
    ++record;
    if (timer.real() > 5 * 1000)
    {
      ostringstream oss;
      oss << "Read " << record << " lines; " << tracks.size() << " tracks";
      if ( file_size > 0)
      {
        oss << vul_sprintf( "; %02.2f%% of file", 100.0*is.tellg()/file_size );
      }
      LOG_INFO( main_logger, oss.str() );
      timer.mark();
    }

    if ( ! p.parse( tmp ))
    {
      LOG_ERROR( main_logger, "Couldn't parse '" << tmp << "'?" );
      return false;
    }

    unsigned external_track_id = static_cast<unsigned>( p.dbl_track_id );

    // initialize the new-track detector at the start of the stream
    bool new_track;
    if ( ! current_external_id_valid )
    {
      current_external_id_valid = true;
      current_external_id = external_track_id;
      new_track = true;
    }
    else
    {
      new_track = ( external_track_id != current_external_id );
    }

    if ( new_track )
    {
      tracks.push_back( kw18.create() );
      kw18.external_id() = external_track_id;
      current_external_id = external_track_id;
    }

    frame_handle_type current_frame = kw18.create_frame();

    kw18[ current_frame ].fg_mask_area() = p.area;
    kw18[ current_frame ].frame_number() = static_cast<unsigned>( p.dbl_frame_num );

    // if timestamp < 0 (usually == -1), do not set!
    if (p.timestamp >= 0)
    {
      kw18[ current_frame ].timestamp_usecs() =
        static_cast<unsigned long long>( p.timestamp * 1000 * 1000 );
    }
    kw18[ current_frame ].track_location() = vgl_point_2d<double>( p.loc_x, p.loc_y );
    kw18[ current_frame ].velocity_x() = p.vel_x;
    kw18[ current_frame ].velocity_y() = p.vel_y;
    kw18[ current_frame ].obj_x() = p.obj_loc_x;
    kw18[ current_frame ].obj_y() = p.obj_loc_y;
    kw18[ current_frame ].obj_location() = vgl_point_2d<double>( p.obj_loc_x, p.obj_loc_y );
    kw18[ current_frame ].bounding_box() =
      vgl_box_2d<double>(
        vgl_point_2d<double>( p.bb_c1_x, p.bb_c1_y ),
        vgl_point_2d<double>( p.bb_c2_x, p.bb_c2_y ));
    kw18[ current_frame ].world_x() = p.world_x;
    kw18[ current_frame ].world_y() = p.world_y;
    kw18[ current_frame ].world_z() = p.world_z;
    kw18[ current_frame ].world_location() = vgl_point_3d<double>( p.world_x, p.world_y, p.world_z );

    // kw19 hacks
    if (this->opts.kw19_hack)
    {
      relevancy( current_frame.row ) = p.kw19;
    }

  } // ... while non-blank lines exist
  return true;

}

bool
file_format_kw18
::write( const string& fn,
         const track_handle_list_type& tracks ) const
{
  ofstream os( fn.c_str() );
  if ( ! os )
  {
    LOG_ERROR( main_logger, "Couldn't open '" << fn << "' for writing" );
    return false;
  }
  this->current_filename = fn;
  bool rc = this->write( os, tracks );
  this->current_filename = "(none)";
  return rc;
}


bool
file_format_kw18
::write( ostream& os,
         const track_handle_list_type& tracks ) const
{
  track_kw18_type kw18;
  map< string, size_t > warnings;

  os << "# 1:track-id 2:track-length 3:frame-number "
     << "4:tracking-plane-loc-x 5:tracking-plane-loc-y "
     << "6:velocity-x 7:velocity-y "
     << "8:image-loc-x 9:image-loc-y "
     << "10:image-bbox-tl-x 11:image-bbox-tl-y 12:image-bbox-br-x 13:image-bbox-br-y "
     << "14:area "
     << "15:world-loc-x 16:world-loc-y 17:world-loc-y "
     << "18:timestamp-secs\n";

  for (size_t i=0; i<tracks.size(); ++i)
  {
    track_handle_type t = tracks[i];
    frame_handle_list_type frames = track_oracle_core::get_frames( t );
    size_t n_frames = frames.size();

    // clunky, but better safe than sorry
    unsigned kw18_id = logged_get_field( kw18.external_id, t.row, warnings );
    for (size_t j=0; j<n_frames; ++j)
    {
      frame_handle_type f = frames[j];

      unsigned long long kw18_ts_usecs = logged_get_field( kw18.timestamp_usecs, f.row, warnings );

      os << kw18.external_id.io_fmt( kw18_id ) << " "
         << n_frames << " "
         << logged_output( kw18.frame_number, f.row, warnings ) << " "
         << logged_output( kw18.track_location, f.row, warnings ) << " "
         << logged_output( kw18.velocity_x, f.row, warnings ) << " "
         << logged_output( kw18.velocity_y, f.row, warnings ) << " "
         << logged_output( kw18.obj_x, f.row, warnings ) << " "
         << logged_output( kw18.obj_y, f.row, warnings ) << " "
         << logged_output( kw18.bounding_box, f.row, warnings ) << " "
         << logged_output( kw18.fg_mask_area, f.row, warnings ) << " "
         << logged_output( kw18.world_x, f.row, warnings ) << " "
         << logged_output( kw18.world_y, f.row, warnings ) << " "
         << logged_output( kw18.world_z, f.row, warnings ) << " ";
      streamsize old_prec = os.precision( 20 );
      os << kw18_ts_usecs / 1.0e6;
      os.precision( old_prec );
      os << "\n";
    }
  }

  if ( ! warnings.empty() )
  {
    size_t sum = 0;
    for (  map< string, size_t >::const_iterator i = warnings.begin();
           i != warnings.end();
           ++i )
    {
      sum += i->second;
      os << "# default values for '" << i->first << "': " << i->second << " times\n";
    }
    LOG_WARN( main_logger, "Writing " << this->current_filename
              << " : used " << sum << " total default values for "
              << warnings.size() << " fields" );
  }

  return true;
}

} // ...track_oracle
} // ...kwiver
