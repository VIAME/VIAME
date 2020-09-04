/*ckwg +5
 * Copyright 2012-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "file_format_vatic.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cctype>

#include <vital/util/tokenize.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::getline;
using std::ifstream;
using std::isdigit;
using std::istream;
using std::set;
using std::string;
using std::stringstream;
using std::vector;

namespace kwiver {
namespace track_oracle {

bool
file_format_vatic
::inspect_file( const string& fn ) const
{
  ifstream is( fn.c_str() );
  string line;
  if (!getline(is, line))
  {
    return false;
  }
  vector <string> fields;

  // quick exit unless quoted strings exist
  if ( line.find( "\"" ) == string::npos ) return false;

  {
    vector< string > fields;
    const bool doTrimEmpty = true;
    kwiver::vital::tokenize( line, fields, " ", doTrimEmpty );
    if (fields.size() < 10  || !isdigit(fields[6][0]))
    {
      return false;
    }
  }


  return true; // Looks enough like a vatic to try and read it.
}

bool
file_format_vatic
::read( const string& fn,
        track_handle_list_type& tracks ) const
{
  ifstream is( fn.c_str() );
  if ( ! is )
  {
    LOG_ERROR( main_logger, "Couldn't read vatic tracks from '" << fn << "'");
    return false;
  }

  return this->read( is, tracks );
}


bool
file_format_vatic
::read( istream& is,
        track_handle_list_type& tracks ) const
{
  track_vatic_type vatic;

  // detect when a new track has started

  bool current_external_id_valid = false;
  dt::tracking::external_id::Type current_external_id = 0;
  //oracle_entry_handle_type track_id;

  unsigned line_count = 0;
  string tmp;
  while ( getline(is, tmp) )
  {
    ++line_count;

    // skip blank lines
    if (tmp.empty())
    {
      continue;
    }

    // matlab always writes doubles
    unsigned int tid; // field 1
    double xmin, ymin, xmax, ymax; // fields 2, 3, 4, 5
    unsigned frame;   // field 6
    bool lost, occluded, generated; // fields 7, 8, 9
    string label; // field 10
    set< string > attributes; // field 11+

    stringstream sstr( tmp );

    if ( ! ( sstr >> tid
                  >> xmin
                  >> ymin
                  >> xmax
                  >> ymax
                  >> frame
                  >> lost
                  >> occluded
                  >> generated ) )
    {
      return false;
    }

    string sep;
    getline(sstr, sep, '"');
    getline(sstr, label, '"');

    // Get the attributes
    {
      string attr;
      while (getline(sstr, sep, '"') && getline(sstr, attr, '"'))
      {
        if (attr.empty())
        {
          continue;
        }

        attributes.insert(attr);
      }
    }

    // initialize the new-track detector at the start of the stream
    bool new_track;
    if ( ! current_external_id_valid )
    {
      current_external_id_valid = true;
      current_external_id = tid;
      new_track = true;
    }
    else
    {
      new_track = ( tid != current_external_id );
    }

    if ( new_track )
    {
      tracks.push_back( vatic.create() );
      vatic.external_id() = tid;
      current_external_id = tid;
    }

    frame_handle_type current_frame = vatic.create_frame();

    vatic[ current_frame ].bounding_box() =
      vgl_box_2d<double>(
        vgl_point_2d<double>( xmin, ymin ),
        vgl_point_2d<double>( xmax, ymax ));

    vatic[ current_frame ].frame_number() = frame;

    vatic[ current_frame ].lost() = lost;
    vatic[ current_frame ].occluded() = occluded;
    vatic[ current_frame ].generated() = generated;

    vatic[ current_frame ].label() = label;

    vatic[ current_frame ].attributes() = attributes;

  } // ...while is.good()

  return true;

}

} // ...track_oracle
} // ...kwiver
