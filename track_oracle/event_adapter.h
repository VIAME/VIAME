// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_EVENT_ADAPTER_H
#define INCL_EVENT_ADAPTER_H

///
/// This class reflects the continually in-flux nature of events in vidtk.
/// I hope that it will eventually subsume aries_interface.
///
/// This class translates vidtk / aries events into track_oracle fields.
/// At the moment, it does NOT read those fields back into events.
///
/// Currently, deals with individual track_fields; perhaps these should be
/// consolidated into a single structure?  If so, how closely should that
/// structure follow vidtk / kwiver events?
///
/// Pending unification of events, only handles VIRAT events (same as ever.)
///

#include <vital/vital_config.h>
#include <track_oracle/track_oracle_event_adapter_export.h>

#include <string>

#include <track_oracle/track_base.h>
#include <track_oracle/track_field.h>

#include <track_oracle/utils/logging_map.h>
#include <track_oracle/track_oracle_api_types.h>

#include <track_oracle/data_terms/data_terms.h>

namespace kwiver {
namespace track_oracle {

struct TRACK_ORACLE_EVENT_ADAPTER_EXPORT event_data_block
{
  bool valid;
  bool debug;
  event_data_block(): valid(false),debug(false) {}
  dt::tracking::external_id::Type new_track_id;
  dt::tracking::external_id::Type src_track_id;
  std::string track_style;

  dt::events::event_id::Type event_id;
  dt::events::event_type::Type event_type;
  dt::events::event_probability::Type event_probability;

  bool has_fn, has_ts;
  dt::tracking::frame_number::Type start_fn, end_fn;
  dt::tracking::timestamp_usecs::Type start_ts, end_ts;

  // given pointers to a frame number and timestamp, either of which
  // (hopefully not both) may be null, decide if that time point is
  // within the (inclusive) ranges of {start,end}_{fn,ts}.  Ideally
  // they're consistent.  If the first bool is false, no decision
  // could be reached.  If the first bool is true, the second is
  // true if the time point is inside the window.
  std::pair< bool, bool > timepoint_in_window( const dt::tracking::frame_number::Type* fn,
                                               const dt::tracking::timestamp_usecs::Type* ts ) const;
};

struct TRACK_ORACLE_EVENT_ADAPTER_EXPORT event_data_schema: public track_base< event_data_schema >
{
  track_field< dt::tracking::external_id > external_id;
  track_field< dt::events::source_track_ids > source_track_ids;
  track_field< dt::events::event_type > activity_id;
  track_field< dt::events::event_probability > activity_probability;
  track_field< dt::events::event_id > event_id;

  event_data_schema()
  {
    Track.add_field( external_id );
    Track.add_field( event_id );
    Track.add_field( activity_id );
    Track.add_field( activity_probability );
    Track.add_field( source_track_ids );
  }
};

class TRACK_ORACLE_EVENT_ADAPTER_EXPORT event_adapter
{
public:

  //
  // given a line from a KWE file, extract the relevant
  // data fields from it.  Returning false signals a
  // serious error (i.e. couldn't parse); returning true
  // but with an invalid event_data_block signals a
  // skip-me-and-continue sort of error (i.e. non-vidtk
  // event.)
  //

  static bool parse_kwe_line( const std::string& line,
                              event_data_block& b,
                              logging_map_type& msgs );

  //
  // Given a new track handle, copy over the data from
  // the event data block
  //

  static void set_event_track_data( const track_handle_type& new_track,
                                    const event_data_block& b,
                                    logging_map_type& msgs );

  //
  // Given a source track handle and an event_data_block,
  // clone the geometry out of the source track into a new
  // track
  //

  static bool clone_geometry( track_handle_type src_track,
                              track_handle_type new_track,
                              const event_data_block& b );

private:
};

} // ...track_oracle
} // ...kwiver

#endif
