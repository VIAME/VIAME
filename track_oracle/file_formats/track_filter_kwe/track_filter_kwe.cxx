/*ckwg +5
 * Copyright 2014-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "track_filter_kwe.h"

#include <cmath>
#include <iostream>
#include <fstream>

#include <kwiversys/RegularExpression.hxx>

#include <vul/vul_awk.h>
#include <vul/vul_timer.h>
#include <vul/vul_sprintf.h>
#include <vul/vul_file.h>

#include <track_oracle/aries_interface/aries_interface.h>
#include <track_oracle/event_adapter.h>
#include <track_oracle/utils/logging_map.h>
#include <track_oracle/data_terms/data_terms.h>


#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::ifstream;
using std::string;
using std::map;
using std::pair;
using std::make_pair;
using std::ostringstream;

namespace kwiver {
namespace track_oracle {

bool
track_filter_kwe_type
::read( const string& fn,
        const track_handle_list_type& ref_tracks,
        const string& track_style,
        track_handle_list_type& new_tracks )
{

  logging_map_type wmsgs( main_logger, KWIVER_LOGGER_SITE );
  ifstream is( fn.c_str() );
  if ( ! is )
  {
    LOG_ERROR( main_logger, "Couldn't open KWE '" << fn << "'" );
    return false;
  }

  // build a local lookup map
  map< unsigned, track_handle_type > id2handle;
  typedef map< unsigned, track_handle_type >::iterator id2handle_it;
  track_field< dt::tracking::external_id > id_field;
  for (size_t i=0; i<ref_tracks.size(); ++i)
  {
    pair< bool, unsigned > probe = id_field.get( ref_tracks[i].row );
    if ( ! probe.first )
    {
      LOG_ERROR( main_logger, "d2d track without external ID?" );
      return false;
    }
    pair< id2handle_it, bool > insert_test = id2handle.insert( make_pair(probe.second, ref_tracks[i]) );
    if ( ! insert_test.second )
    {
      LOG_ERROR( main_logger, "Duplicate track IDs " << probe.second );
      return false;
    }
  }

  string line;
  size_t nlines = 0;
  kwiversys::RegularExpression comment_re( "^ *#" );
  vul_timer timer;
  size_t file_size = vul_file::size( fn );
  while (getline( is, line ))
  {
    if (timer.real() > 5 * 1000)
    {
      ostringstream oss;
      oss << "Read " << nlines << " lines; " << new_tracks.size() << " events";
      oss << vul_sprintf( "; %02.2f%% of file", 100.0*is.tellg()/file_size );
      LOG_INFO( main_logger, oss.str() );
      timer.mark();
    }

    if ( comment_re.find( line )) continue;

    event_data_block b;
    if ( ! event_adapter::parse_kwe_line( line, b, wmsgs ))  return false;
    if ( ! b.valid ) continue;

    id2handle_it probe = id2handle.find( b.src_track_id );
    if ( probe == id2handle.end() )
    {
      // track may have been filtered out
      continue;
    }

    b.debug = (new_tracks.size() < 5);
    b.track_style = track_style;

    track_handle_type new_track( track_oracle_core::get_next_handle() );

    // do this first to correctly set track_style
    if ( ! track_oracle_core::clone_nonsystem_fields( probe->second, new_track ))
    {
      LOG_ERROR( main_logger, "Couldn't clone non-system track level fields?" );
      continue;
    }

    // now we can copy the track_style over
    event_adapter::set_event_track_data( new_track, b, wmsgs );

    // ...and the frame data
    if ( ! event_adapter::clone_geometry( probe->second, new_track, b )) return false;

    new_tracks.push_back( new_track );
    ++nlines;
  }

  if ( ! wmsgs.empty() )
  {
    LOG_INFO( main_logger, "Skipped KWE events: ");
    wmsgs.dump_msgs();
  }

  return true;
}

} // ...track_oracle
} // ...kwiver
