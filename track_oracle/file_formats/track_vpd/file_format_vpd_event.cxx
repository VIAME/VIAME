// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "file_format_vpd_event.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <ctype.h>

#include <vgl/vgl_box_2d.h>

#include <vital/util/string.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::getline;
using std::ifstream;
using std::istream;
using std::istringstream;
using std::map;
using std::ostringstream;
using std::sscanf;
using std::string;
using std::vector;

namespace {

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

struct vpd_event_line_parser
{
  unsigned event_id, event_type, duration, start_frame, end_frame, current_frame;
  double box_lefttop_x, box_lefttop_y, box_width, box_height;

  bool parse( const string& s )
  {
    return (sscanf( s.c_str(),
                        "%d %d %d %d %d %d %lf %lf %lf %lf",
                        &this->event_id,
                        &this->event_type,
                        &this->duration,
                        &this->start_frame,
                        &this->end_frame,
                        &this->current_frame,
                        &this->box_lefttop_x,
                        &this->box_lefttop_y,
                        &this->box_width,
                        &this->box_height ) == 10);
  }
};

struct vpd_mapping_line_parser
{
  unsigned event_id, event_type, event_duration, start_frame, end_frame, n_objects;
  vector< unsigned > object_id_list;

  bool parse( const string& s )
  {
    istringstream iss( s );
    if (! ( iss >> event_id >> event_type >> event_duration >> start_frame >> end_frame >> n_objects ))
    {
      LOG_ERROR( main_logger, "VPD_EVENT: Mapping parser: couldn't parse preamble in '" << s << "'" );
      return false;
    }
    unsigned object_id = 0;
    unsigned object_involved_flag = 0;
    while ( iss >> object_involved_flag )
    {
      if (object_involved_flag)
      {
        object_id_list.push_back( object_id );
      }
      ++object_id;
    }
    if (object_id_list.size() != n_objects)
    {
      LOG_WARN( main_logger, "VPD_EVENT: Mapping parser: line '" << s << "' specifies "
                << n_objects << " objects but has states for " << object_id_list.size() );
    }

    return true;
  }
};

string
mapping_fn_from_event_fn( const string& s )
{
  string mapping_fn( s );
  string key( ".events.txt" );
  size_t p = mapping_fn.rfind( key );
  if ( p == string::npos )
  {
    // don't need to log an error here; open_event_and_mapping_streams handles it
    return "";
  }
  mapping_fn.replace( p, key.length(), ".mapping.txt" );
  return mapping_fn;
}

// return error messages in a string so we can quietly call this
// for inspection

bool
open_event_and_mapping_streams( const string& event_fn,
                                ifstream& event_is,
                                ifstream& mapping_is,
                                string& errors )
{
  string mapping_fn = mapping_fn_from_event_fn( event_fn );
  ostringstream oss;
  if ( mapping_fn.empty() )
  {
    oss << "No mapping filename could be deduced from '" << event_fn << "'";
    errors = oss.str();
    return false;
  }

  event_is.open( event_fn.c_str() );
  if ( ! event_is )
  {
    oss << "Couldn't open event file '" << event_fn << "'";
    errors = oss.str();
    return false;
  }

  mapping_is.open( mapping_fn.c_str() );
  if ( ! mapping_is )
  {
    oss << "Couldn't open mapping file '" << mapping_fn << "'";
    errors = oss.str();
    return false;
  }

  errors = "";
  return true;
}

} // anon namespace

namespace kwiver {
namespace track_oracle {

bool
file_format_vpd_event
::inspect_file( const string& event_fn ) const
{
  ifstream event_is, mapping_is;
  string errors;
  if ( ! open_event_and_mapping_streams( event_fn, event_is, mapping_is, errors ))
  {
    return false;
  }

  // Either both files are empty, or both are non-empty
  string event_line, mapping_line;
  bool event_nonblank = get_next_nonblank_line( event_is, event_line );
  bool mapping_nonblank = get_next_nonblank_line( mapping_is, mapping_line );

  if ( (! event_nonblank) && (! mapping_nonblank ))
  {
    // both lines are blank; that's okay
    return true;
  }

  // at least one non-blank line; both have to parse
  vpd_event_line_parser event_p;
  vpd_mapping_line_parser mapping_p;
  return (event_p.parse( event_line ) && mapping_p.parse( mapping_line ));
}

bool
file_format_vpd_event
::read( const string& event_fn,
        track_handle_list_type& events ) const
{
  ifstream event_is, mapping_is;
  string errors;
  if ( ! open_event_and_mapping_streams( event_fn, event_is, mapping_is, errors ))
  {
    LOG_ERROR( main_logger, "VPD_EVENT: " << errors );
    return false;
  }

  string tmp;
  // load up the mapping data
  map< unsigned, vpd_mapping_line_parser > mappings; // key = event_id
  while (get_next_nonblank_line( mapping_is, tmp ))
  {
    vpd_mapping_line_parser p;
    if (p.parse( tmp ))
    {
      if (mappings.find( p.event_id ) != mappings.end() )
      {
        LOG_ERROR( main_logger, "VPD_EVENT: mapping for '" << event_fn << "': duplicate event ID " << p.event_id );
        return false;
      }
      mappings[ p.event_id ] = p;
    }
    else
    {
      LOG_ERROR( main_logger, "VPD_EVENT: Couldn't parse mapping line '" << tmp << "'" );
      return false;
    }
  }

  //
  // Loop over each line in the event file, which contains one line per
  // frame of the event. Note that we only record the track IDs here and
  // don't assert any relationship between the event boxes and the track boxes
  // or anything else like that. (Even if we wanted to, we couldn't, since
  // we don't have the tracks here.)
  //

  vpd_event_line_parser event_p;
  track_vpd_event_type vpd;
  bool current_event_id_valid = false;
  unsigned current_event_id = 0;

  while (get_next_nonblank_line( event_is, tmp ))
  {
    if ( ! event_p.parse( tmp ))
    {
      LOG_ERROR( main_logger, "VPD_EVENT: file '" << event_fn << "': couldn't parse '" << tmp << "'" );
      return false;
    }

    bool new_event = false;
    if ( ! current_event_id_valid )
    {
      current_event_id = event_p.event_id;
      current_event_id_valid = true;
      new_event = true;
    }
    else
    {
      new_event = (event_p.event_id != current_event_id);
    }

    if (new_event)
    {
      // verify the event was mentioned in the mapping
      map< unsigned, vpd_mapping_line_parser >::const_iterator m = mappings.find( event_p.event_id );
      if ( m == mappings.end())
      {
        LOG_ERROR( main_logger, "VPD_EVENT: Event file '" << event_fn << "': event " << event_p.event_id << " not in mapping file" );
        return false;
      }
      // verify event, start_frame, end_frame
      string mismatch_fields = "";
      if (m->second.event_type != event_p.event_type ) mismatch_fields += " event_type ";
      if (m->second.start_frame != event_p.start_frame ) mismatch_fields += " start_frame ";
      if (m->second.end_frame != event_p.end_frame) mismatch_fields += " end_frame ";
      if (! mismatch_fields.empty())
      {
        LOG_ERROR( main_logger, "VPD_EVENT: Event file '" << event_fn << "': event vs. mapping mismatch in fields " << mismatch_fields );
        return false;
      }

      // It all looks good; create the new event
      events.push_back( vpd.create() );
      vpd.event_id() = event_p.event_id;
      vpd.event_type() = event_p.event_type;
      vpd.start_frame() = event_p.start_frame;
      vpd.end_frame() = event_p.end_frame;
      vpd.object_id_list() = m->second.object_id_list;

      current_event_id = event_p.event_id;
    }

    frame_handle_type frame = vpd.create_frame();
    vpd[ frame ].frame_number() = event_p.current_frame;
    vpd[ frame ].bounding_box() =
      vgl_box_2d<double>( event_p.box_lefttop_x, event_p.box_lefttop_x + event_p.box_width,
                          event_p.box_lefttop_y, event_p.box_lefttop_y + event_p.box_height );

  } // for each non-blank event file line

  return true;
}

} // ...track_oracle
} // ...kwiver
