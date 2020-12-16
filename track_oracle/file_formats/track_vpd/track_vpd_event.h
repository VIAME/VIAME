// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_TRACK_VPD_EVENT_H
#define INCL_TRACK_VPD_EVENT_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_vpd/track_vpd_export.h>

#include <track_oracle/core/track_base.h>
#include <track_oracle/core/track_field.h>
#include <track_oracle/data_terms/data_terms.h>

#include <string>
#include <utility>
#include <vgl/vgl_box_2d.h>

namespace kwiver {
namespace track_oracle {

/// This is the track_oracle schema for the VIRAT Public Data 2.0
/// event tracks.

struct TRACK_VPD_EXPORT track_vpd_event_type: public track_base< track_vpd_event_type >
{
  track_field< dt::events::event_id > event_id;
  track_field< unsigned >& event_type;
  track_field< unsigned >& start_frame;
  track_field< unsigned >& end_frame;
  track_field< std::vector< unsigned > >& object_id_list;
  track_field< dt::tracking::frame_number > frame_number;
  track_field< dt::tracking::bounding_box > bounding_box;
  track_vpd_event_type():
    event_type( Track.add_field< unsigned >( "vpd_event_type" )),
    start_frame( Track.add_field< unsigned >( "start_frame" )),
    end_frame( Track.add_field< unsigned >( "end_frame" )),
    object_id_list( Track.add_field< std::vector< unsigned > >( "object_id_list" ))
  {
    Track.add_field( event_id );
    Frame.add_field( frame_number );
    Frame.add_field( bounding_box );
  }

  static std::string event_type_to_str( unsigned t );
  static unsigned str_to_event_type( const std::string& s );
};

} // ...track_oracle
} // ...kwiver

#endif
