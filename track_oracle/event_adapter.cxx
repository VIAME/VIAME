// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "event_adapter.h"

#include <cmath>

#include <sstream>

#include <vul/vul_awk.h>

#include <track_oracle/aries_interface/aries_interface.h>
#include <track_oracle/data_terms/data_terms.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::pair;
using std::stringstream;
using std::ostringstream;
using std::vector;
using std::make_pair;
using std::string;

namespace kwiver {
namespace track_oracle {

pair<bool, bool>
event_data_block
::timepoint_in_window( const dt::tracking::frame_number::Type* fn,
                       const dt::tracking::timestamp_usecs::Type* ts ) const
{
  if ( ! this->valid )
  {
    LOG_ERROR( main_logger, "Logic error: timepoint_in_window on an invalid data block" );
    return make_pair( false, false );
  }
  if ( ! ( this->has_fn || this->has_ts ))
  {
    LOG_ERROR( main_logger, "Logic error: event data block marked valid but has no timestamp / frame number?" );
    return make_pair( false, false );
  }
  if ( ! ( fn || ts ))
  {
    LOG_ERROR( main_logger, "Logic error: timepoint_in_window called with null time data" );
    return make_pair( false, false );
  }

  pair< bool, bool > ret( false, false );
  // timestamp first, so frame number can override in case of conflict
  if ( this->has_ts && ts )
  {
    ret.second = ( this->start_ts <= *ts ) && ( *ts <= this->end_ts );
    ret.first = true;
  }
  if ( this->has_fn && fn )
  {
    bool fn_check = ( this->start_fn <= *fn ) && ( *fn <= this->end_fn );
    if ( ret.first && ( ret.second != fn_check ))
    {
      LOG_WARN( main_logger, "event_adapter: timepoint in window: timestamp check for " << *ts
                << " says " << ret.second << "; framenumber check for " << *fn
                << " says " << fn_check << "; going with framenumber" );
    }
    ret.first = true;
    ret.second = fn_check;
  }
  return ret;
}

bool
event_adapter
::parse_kwe_line( const string& line,
                  event_data_block& b,
                  logging_map_type& msgs )
{
  b.valid = false;

  stringstream line_ss( line );
  vul_awk awk( line_ss );
  if ( awk.NF() < 14)
  {
    LOG_ERROR( main_logger, "Expected at least 14 tokens; found in " << awk.NF() << " in '" << line << "'" );
    return false;
  }
  stringstream ss( string(awk[2]) + " " + awk[13] + " " + awk[4] + " " + awk[6] + " "
                   + awk[3] + " " + awk[5] + " "+ awk[1] + " " + awk[7] );
  int vidtk_event_type;
  if ( ! ( ss
           >> b.event_id
           >> b.src_track_id
           >> b.start_fn
           >> b.end_fn
           >> b.start_ts
           >> b.end_ts
           >> vidtk_event_type
           >> b.event_probability
         ))
  {
    LOG_ERROR( main_logger, "Couldn't parse CSV line '" << line << "'" );
    return false;
  }

  // KWEs get read into their own track lists, so no need to bias the track IDs
  b.new_track_id = b.event_id;

  b.has_fn = true;
  b.has_ts = true;

  if ((b.start_ts > 0) && (log( static_cast<double>(b.start_ts) ) < 28.0 ))
  {
    LOG_WARN( main_logger, "Warning: KWE start timestamp '" << b.start_ts << "' may not be in usecs" );
  }
  if ((b.end_ts > 0) && (log( static_cast<double>(b.end_ts) ) < 28.0 ))
  {
    LOG_WARN( main_logger, "Warning: KWE end timestamp '" << b.end_ts << "' may not be in usecs" );
  }

  vidtk::event_types::enum_types kwe_event_type = static_cast<vidtk::event_types::enum_types>( vidtk_event_type );
  string virat_act_name = aries_interface::kwe_index_to_activity( kwe_event_type );
  if ( virat_act_name == "" )
  {
    ostringstream oss;
    oss << "No VIRAT correspondence for KWE activity " << vidtk_event_type;
    msgs.add_msg( oss.str() );
  }
  else
  {
    b.event_type = aries_interface::activity_to_index( virat_act_name );
    b.valid = true;
  }
  return true;
}

void
event_adapter
::set_event_track_data( const track_handle_type& new_track,
                        const event_data_block& b,
                        logging_map_type& msgs )
{

  event_data_schema eds;

  // set track-level data
  // (the event / activity confusion is regrettable, but
  // correctly mirrors the real world)

  eds( new_track ).external_id() = b.new_track_id;
  vector<unsigned> src_ids;
  src_ids.push_back( b.src_track_id );
  eds( new_track ).source_track_ids() = src_ids;
  eds( new_track ).activity_id() = b.event_type;
  eds( new_track ).activity_probability() = b.event_probability;
  eds( new_track ).event_id() = b.event_id;

  // also set probabilities as a descriptor
  track_field< dt::virat::descriptor_classifier > descriptor_classifier;
  vector<double> dc( aries_interface::index_to_activity_map().size() );
  dc[ b.event_type ] = b.event_probability;
  descriptor_classifier( new_track.row ) = dc;

  // only set the track style if it's specified
  {
    track_field< dt::tracking::track_style > track_style_field;
    if ( b.track_style != "" )
    {
      track_style_field( new_track.row ) = b.track_style;
      msgs.add_msg( "event_adapter: setting track style to '"+b.track_style+"'" );
    }
    else
    {
      pair< bool, string > track_style_probe = track_style_field.get( new_track.row );
      if ( track_style_probe.first )
      {
        msgs.add_msg( "event_adapter: no track style specified; using style '"+track_style_probe.second+"' from source" );
      }
      else
      {
        msgs.add_msg( "event_adapter: no track style specified; none found in source" );
      }
    }
  }
}

bool
event_adapter
::clone_geometry( track_handle_type src_track,
                  track_handle_type new_track,
                  const event_data_block& b )
{

  // copy over those frames in the time window
  frame_handle_list_type frames = track_oracle_core::get_frames( src_track );
  track_field< dt::tracking::frame_number > fn_field;
  track_field< dt::tracking::timestamp_usecs > ts_field;
  event_data_schema eds;

  for (size_t i=0; i<frames.size(); ++i)
  {
    frame_handle_type f = frames[i];

    pair< bool, dt::tracking::frame_number::Type > fn_probe = fn_field.get( f.row );
    pair< bool, dt::tracking::timestamp_usecs::Type > ts_probe = ts_field.get( f.row );

    dt::tracking::frame_number::Type* fn_ptr =
      fn_probe.first
      ? &fn_probe.second
      : 0;
    dt::tracking::timestamp_usecs::Type* ts_ptr =
      ts_probe.first
      ? &ts_probe.second
      : 0;

    pair< bool, bool > check = b.timepoint_in_window( fn_ptr, ts_ptr );
    if ( ! check.second )
    {
      continue;
    }

    // create a frame and clone everything over
    frame_handle_type new_frame = eds( new_track ).create_frame();
    if ( ! track_oracle_core::clone_nonsystem_fields( f, new_frame ) )
    {
      return false;
    }

  } // for each frame

  if (b.debug)
  {
    size_t nf = track_oracle_core::get_n_frames( new_track );
    LOG_DEBUG( main_logger, "Created track of activity " << b.event_type << " prob " <<
               b.event_probability << " with " << nf << " frames" );
  }

  return true;
}

} // ...track_oracle
} // ...kwiver
