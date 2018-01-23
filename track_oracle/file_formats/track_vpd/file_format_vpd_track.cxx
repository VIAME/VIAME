/*ckwg +5
 * Copyright 2013-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "file_format_vpd_track.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <ctype.h>

#include <vul/vul_awk.h>

#include <vgl/vgl_box_2d.h>

#include <vital/util/string.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::getline;
using std::ifstream;
using std::istream;
using std::istringstream;
using std::sscanf;
using std::string;

namespace { // anon

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

struct vpd_track_line_parser
{
  unsigned object_id, object_duration, current_frame;
  double box_lefttop_x, box_lefttop_y, box_width, box_height;
  unsigned object_type;

  bool parse( const string& s )
  {
    istringstream iss( s );
    vul_awk awk( iss );
    if (awk.NF() != 8) return false;
    return (sscanf( s.c_str(),
                        "%d %d %d %lf %lf %lf %lf %d",
                        &this->object_id,
                        &this->object_duration,
                        &this->current_frame,
                        &this->box_lefttop_x,
                        &this->box_lefttop_y,
                        &this->box_width,
                        &this->box_height,
                        &this->object_type ) == 8);
  }
};


} // anon namespace

namespace kwiver {
namespace track_oracle {

bool
file_format_vpd_track
::inspect_file( const string& fn ) const
{
  ifstream is( fn.c_str() );
  if ( ! is )
  {
    LOG_ERROR( main_logger, "Couldn't open '" << fn << "'" );
    return false;
  }

  string line;
  if ( ! get_next_nonblank_line( is, line )) return false;
  vpd_track_line_parser p;
  return p.parse( line );
}

bool
file_format_vpd_track
::read( const string& fn,
        track_handle_list_type& tracks ) const
{
  ifstream is( fn.c_str() );
  if ( ! is )
  {
    LOG_ERROR( main_logger, "Couldn't open '" << fn << "'" );
    return false;
  }

  return this->read( is, tracks );
}


bool
file_format_vpd_track
::read( istream& is,
        track_handle_list_type& tracks ) const
{
  track_vpd_track_type vpd;
  string line;
  vpd_track_line_parser p;

  bool current_object_id_valid = false;
  unsigned current_object_id = 0;

  while ( get_next_nonblank_line( is, line ))
  {
    if ( ! p.parse( line ))
    {
      LOG_ERROR( main_logger, "Couldn't parse '" << line << "'?" );
      return false;
    }

    bool new_track = false;
    if ( ! current_object_id_valid )
    {
      current_object_id = p.object_id;
      current_object_id_valid = true;
      new_track = true;
    }
    else
    {
      new_track = ( p.object_id != current_object_id );
    }

    if ( new_track )
    {
      tracks.push_back( vpd.create() );
      vpd.object_id() = p.object_id;
      vpd.object_type() = p.object_type;
      current_object_id = p.object_id;
    }

    frame_handle_type frame = vpd.create_frame();
    vpd[ frame ].frame_number() = p.current_frame;
    vpd[ frame ].bounding_box() =
      vgl_box_2d<double>( p.box_lefttop_x, p.box_lefttop_x + p.box_width,
                          p.box_lefttop_y, p.box_lefttop_y + p.box_height );

  } // while non-blank lines remain
  return true;
}

} // ...track_oracle
} // ...kwiver
